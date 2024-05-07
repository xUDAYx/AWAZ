import asyncio
import datetime
import logging
import os,sys
import time
import traceback
from argparse import ArgumentParser
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)
logging.basicConfig(level=logging.WARNING)
import assets.themes.loadThemes as loadThemes
from pydub import AudioSegment
import soundfile as sf
import edge_tts
import gradio as gr
import librosa
import torch
from fairseq import checkpoint_utils

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC


#######################################################################################################
#                                      DIR SETUP                                                      #
#######################################################################################################

warnings.filterwarnings("ignore", category=UserWarning, module='gradio.mix')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
now_dir = os.getcwd()
sys.path.append(now_dir)
rvc_models_dir = os.path.join(BASE_DIR, 'weights')
bg_music= os.path.join(now_dir, "assets", "bg_music")
voice_sample = os.path.join(now_dir, "assets", "voice_sample")
os.makedirs(voice_sample, exist_ok=True)
sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}
bg_music_relative = os.path.relpath(bg_music, now_dir)
bg_music_files = [f for f in os.listdir(bg_music) if f.endswith(tuple(sup_audioext))]


#######################################################################################################
#                                      DO NOT TOUCH                                                   #
#######################################################################################################

# logging.getLogger("fairseq").setLevel(logging.WARNING)
# logging.getLogger("numba").setLevel(logging.WARNING)
# logging.getLogger("markdown_it").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("matplotlib").setLevel(logging.WARNING)

limitation = os.getenv("SYSTEM") == "spaces"

config = Config()

edge_output_filename = "TTS_output.mp3"
tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]

model_root = "weights"
models = [
    d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
]
if len(models) == 0:
    raise ValueError("No model found in `weights` folder")
models.sort()


def model_data(model_name):
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".pth")
    ]
    if len(pth_files) == 0:
        raise ValueError(f"No pth file found in {model_root}/{model_name}")
    pth_path = pth_files[0]
    # print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    # print("Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    # n_spk = cpt["config"][-3]

    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        # print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        # print(f"Index file found: {index_file}")

    return tgt_sr, net_g, vc, version, index_file, if_f0


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


# print("Loading hubert model...")
hubert_model = load_hubert()
# print("Hubert model loaded.")

# print("Loading rmvpe model...")
rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)
# print("rmvpe model loaded.")


#######################################################################################################
#                                 Combine Audio Funtion                                               #
#######################################################################################################
def combine_audio(bg_music, volume_slider): # bg_music is file name for bg music file 
        # bg_music_path = os.path.join(now_dir, "assets", "bg_music") # path/bg_music
        # temp = "assets/bg_music/" + bg_music
        # voice_sample_audio = AudioSegment.from_file(temp)
        # reduced_audio = voice_sample_audio - volume_slider
        # bg_music = reduced_audio 

        bg_music = AudioSegment.from_file(f"assets/bg_music/{bg_music}") - volume_slider
            
        # Get the length of voice_sample
        # voice_sample_path = os.path.join(now_dir, "assets", "voice_sample","cloned_voice.wav")
        voice_sample = AudioSegment.from_file("assets/voice_sample/cloned_voice.wav")
        voice_sample_length = len(voice_sample)

        # Check if bg_music is shorter than voice_sample
        if len(bg_music) < voice_sample_length:
            # Repeat bg_music to match the length of voice_sample
            repeated_bg_music = bg_music * (voice_sample_length // len(bg_music)) + bg_music[:voice_sample_length % len(bg_music)]
            bg_music_trimmed = repeated_bg_music
        else:
            # Trim bg_music to the length of voice_sample
            bg_music_trimmed = bg_music[:voice_sample_length]
        combined_audio = bg_music_trimmed.overlay(voice_sample)
        file_path = "output.mp3"
        combined_audio.export(file_path, format="mp3")
        return file_path

#######################################################################################################
#                                      TTS TAB FUNTION                                                #
#######################################################################################################

def tts(
    model_name,
    speed,
    tts_text,
    tts_voice,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
):
    print("------------------------------------")
    print(datetime.datetime.now())
    # print("TTS_Text:",tts_text)
    print(f"TTS_Voice: {tts_voice}")
    print(f"Model name: {model_name}")
    # print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")
    try:
        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        t0 = time.time()
        if speed >= 0:
            speed_str = f"+{speed}%"
        else:
            speed_str = f"{speed}%"
        asyncio.run(
            edge_tts.Communicate(
                tts_text, "-".join(tts_voice.split("-")[:-1]), rate=speed_str
            ).save(edge_output_filename)
        )
        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")

        f0_up_key = int(f0_up_key)

        if not hubert_model:
            load_hubert()
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            edge_output_filename,
            times,
            f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        tts_audio = audio_opt
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr

        # Save tts_audio to a WAV file
        sf.write(os.path.join(voice_sample, "cloned_voice.wav"), audio_opt, tgt_sr)

        # info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        info="Succefully Coloned Voice"
        print(info)
        return (
            info,
            edge_output_filename,
            (tgt_sr, audio_opt),
            tts_audio,
        )
    
    except EOFError:
        info = (
            "It seems that the edge-tts output is not valid. "
            "This may occur when the input text and the speaker do not match. "
            "For example, maybe you entered Japanese (without alphabets) text but chose non-Japanese speaker?"
        )
        print(info)
        return info, None, None
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None


#######################################################################################################
#                                      Gradio UI TTS TAB                                              #
#######################################################################################################

def tts_tab():
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(label="Model", choices=models, value=models[0])
                tts_voice = gr.Dropdown(
                    label="TTS speaker",
                    choices=tts_voices,
                    allow_custom_value=False,
                    value="hi-IN-SwaraNeural-Female",
                )
                f0_method = gr.Radio(
                    label="Pitch extraction method (Rmvpe is default)",
                    choices=["rmvpe"],  # harvest is too slow
                    value="rmvpe",
                    interactive=True,
                )
                
                
            with gr.Column():
                
                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Index rate",
                    value=1,
                    interactive=True,
                )
                f0_key_up = gr.Slider(
                    minimum=-15,
                    maximum=15,
                    step=1,
                    label="Pitch",
                    value=0,
                )
                speed = gr.Slider(
                    minimum=-100,
                    maximum=100,
                    label="Speech speed (%)",
                    value=0,
                    step=10,
                    interactive=True,
                )  
                
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Protect",
                    value=0.33,
                    step=0.01,
                    interactive=True,
                )
        with gr.Row():
            with gr.Column():
                tts_text = gr.Textbox(label="Input Text", 
                                      value="दिल्ली के जाने माने व्यापारी। सेठ मनोहरदास जी उनके एक बहुत खास मित्र थे जानकीदास। अपने व्यापार की हर सलाह वे जानकीदास से लेते थे।",
                                      lines=15) 
                # txt_file = gr.File(
                #         label=("Or you can upload a .txt file"),
                #         type="file",
                #     )
                # txt_file.upload(
                #    fn=process_input,
                #    inputs=[txt_file],
                #    outputs=[tts_text],
                # )

            with gr.Column():

                info_text = gr.Textbox(label="Output info")
                edge_tts_output = gr.Audio(label="TTS Voice", type="filepath")
                tts_output = gr.Audio(label="Cloned Voice")
        with gr.Column():
            but0 = gr.Button("Convert", variant="primary")

            but0.click(
                tts,
                [
                    model_name,
                    speed,
                    tts_text,
                    tts_voice,
                    f0_key_up,
                    f0_method,
                    index_rate,
                    protect0,
                ],
                [info_text, edge_tts_output, tts_output],
            )

        with gr.Row():
            bg_music = gr.Dropdown(
                label=("Background Audio"),
                info=("Select the Background Music."),
                choices= bg_music_files,
                value="bg_2.mp3",
                interactive=True,)
            volume_slider = gr.Slider(
                    minimum=0,
                    maximum=30,
                    label=("Volume"),
                    info="Adjust the volume of the background music. Lower values (e.g. 0) will result in louder music, while higher values (e.g. 30) will result in softer music.",
                    value=15,
                    step=1,
                    interactive=True,
                ) 
        with gr.Column():    
            combine_output = gr.Audio(label="Combined Audio")
        with gr.Column():
                but1 = gr.Button("Combine", variant="primary")
                but1.click(
                    combine_audio,[
                        bg_music,
                        volume_slider
                        ],
                        outputs=[combine_output]
                )







        # with gr.Row():
        #     examples = gr.Examples(
        #         examples_per_page=100,
        #         examples=[
        #             ['''これは日本語テキストから音声への変換デモです。''', "ja-JP-NanamiNeural-Female"],
        #             [
        #                 "This is an English text to speech conversation demo.",
        #                 "en-US-AriaNeural-Female",
        #             ],
        #         ],
        #         inputs=[tts_text, tts_voice , tts_output],
        #     )

              


#######################################################################################################
#                                          APP LUANCH                                                 #
#######################################################################################################


my_app = loadThemes.load_json()
if my_app:
    pass
else:
    my_app = "ParityError/Interstellar"

with gr.Blocks(theme=my_app, title="AWAZ") as app:
    gr.Markdown("## FINAL YEAR PROJECT")
    gr.Markdown(
        (
            "This project is a prototype for a personalized audio book maker. It transforms text into lifelike speech, enabling users to personalize their voice. Please note that it's a prototype and will evolve into an application-based platform, mimicking the user's voice to create personalized audio books."
        )
    )  # Consistent variable name 'app'
    # with gr.Tab("AUDIO BOOK MAKER"):
    #     audio_book_tab()
    
    with gr.Tab("AUDIO BOOK MAKER"):
        tts_tab()

app.launch()