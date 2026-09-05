import os
import json
import time
import threading
import random
import re
import requests
import wave
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

import script_tts_loop as tts


# =========================================================
# 基本目录
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = BASE_DIR / "scripts"
OUTPUT_DIR = BASE_DIR / "output"

app = FastAPI(title="AI直播控制台")


# =========================================================
# 播放器状态
# =========================================================

play_lock = threading.Lock()
play_thread: Optional[threading.Thread] = None

stop_event = threading.Event()
pause_event = threading.Event()

state_lock = threading.Lock()

player_state = {
    "status": "stopped",

    "product": "",

    "current_index": 0,
    "total": 0,

    "current_title": "",
    "current_text": "",

    "current_version": "",

    "current_audio": "",
    "current_audio_duration": 0,

    "round_no": 0,

    "progress_percent": 0,

    "live_started_at": None,
    "live_stopped_at": None,

    "alert_level": "ok",
    "alert_message": "",

    "message": "等待开始直播",

    "logs": [],
}


# =========================================================
# TTS 后台生成状态
# =========================================================

tts_lock = threading.Lock()
tts_thread: Optional[threading.Thread] = None

tts_state_lock = threading.Lock()

tts_state = {
    "status": "idle",
    # idle / loading / generating / completed / error

    "product": "",

    "current_index": 0,
    "total": 0,
    "percent": 0,

    "current_tag": "",
    "current_version": "",

    "message": "等待生成",

    "error": "",
}


# =========================================================
# DeepSeek AI 话术生成状态
# =========================================================

ai_lock = threading.Lock()
ai_thread: Optional[threading.Thread] = None

ai_state_lock = threading.Lock()

ai_state = {
    "status": "idle",
    # idle / generating / completed / error

    "product": "",

    "message": "等待生成",

    "error": "",

    "current": 0,
    "total": 0,
    "percent": 0,

    "current_style": "",
}


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

DEEPSEEK_MODEL = "deepseek-v4-flash"


# =========================================================
# CosyVoice 模型
# =========================================================

cosyvoice = None

model_lock = threading.Lock()


# =========================================================
# 请求模型
# =========================================================

class ProductRequest(BaseModel):
    product: str
    style: str = "高转化直播型"


class ProductCreateRequest(BaseModel):
    product: str
    script: str = ""

    price: str = ""
    specification: str = ""

    selling_points: str = ""

    target_people: str = ""

    usage_scene: str = ""

    notes: str = ""

    extra: str = ""


class ProductInfoRequest(BaseModel):
    product: str

    price: str = ""

    specification: str = ""

    selling_points: str = ""

    target_people: str = ""

    usage_scene: str = ""

    notes: str = ""

    extra: str = ""


class LocalOpenRequest(BaseModel):
    product: str
    kind: str


# =========================================================
# 播放器状态
# =========================================================

def set_state(**kwargs):
    with state_lock:
        player_state.update(kwargs)


def get_state():

    with state_lock:

        state = dict(player_state)

        state["logs"] = list(
            player_state.get("logs", [])
        )

        return state


def add_live_log(message: str):

    text = str(message or "").strip()

    if not text:
        return

    stamp = time.strftime("%H:%M:%S")

    with state_lock:

        logs = player_state.setdefault(
            "logs",
            []
        )

        logs.append(
            f"[{stamp}] {text}"
        )

        if len(logs) > 150:
            del logs[:-150]


# =========================================================
# TTS 状态
# =========================================================

def set_tts_state(**kwargs):

    with tts_state_lock:
        tts_state.update(kwargs)


def get_tts_state():

    with tts_state_lock:
        return dict(tts_state)


# =========================================================
# DeepSeek 状态
# =========================================================

def set_ai_state(**kwargs):

    with ai_state_lock:
        ai_state.update(kwargs)


def get_ai_state():

    with ai_state_lock:
        return dict(ai_state)


# =========================================================
# WAV 时长
# =========================================================

def get_wav_duration(wav_file):

    try:

        with wave.open(
            str(wav_file),
            "rb"
        ) as wf:

            frames = wf.getnframes()

            rate = wf.getframerate()

            if not rate:
                return 0

            return round(
                frames / rate,
                2
            )

    except Exception:

        return 0


# =========================================================
# 商品
# =========================================================

def validate_product_name(product: str):

    product = (
        product or ""
    ).strip()

    if not product:

        raise RuntimeError(
            "商品名称不能为空。"
        )

    if len(product) > 80:

        raise RuntimeError(
            "商品名称不能超过80个字符。"
        )

    if product in {
        ".",
        ".."
    } or re.search(
        r'[\\/:*?"<>|]',
        product
    ):

        raise RuntimeError(
            "商品名称不能包含 \\ / : * ? \" < > | 等特殊字符。"
        )

    return product


def default_product_script(product: str):

    return (
        f"""[开场]\n"""
        f"""大家好，今天给大家分享一下{product}。\n\n"""
        f"""[产品介绍]\n"""
        f"""这款商品的具体信息，请大家以商品页面和实际产品为准。\n\n"""
        f"""[产品卖点]\n"""
        f"""它的主要特点和优势，大家可以结合页面介绍了解一下。\n\n"""
        f"""[使用方法]\n"""
        f"""具体使用方法请按照商品包装或说明书上的要求操作。\n\n"""
        f"""[促单]\n"""
        f"""如果这款商品正好符合你的需求，可以先了解一下，再决定是否购买。\n\n"""
        f"""[结束]\n"""
        f"""今天的分享就到这里，感谢大家的观看。\n"""
    )


PRODUCT_INFO_FILE = "product_info.json"


def product_info_file(product: str) -> Path:

    return (
        SCRIPT_DIR /
        product /
        PRODUCT_INFO_FILE
    )


def default_product_info(product: str):

    return {

        "product":
            product,

        "price":
            "",

        "specification":
            "",

        "selling_points":
            "",

        "target_people":
            "",

        "usage_scene":
            "",

        "notes":
            "",

        "extra":
            "",
    }


def get_product_info(product: str):

    product = validate_product_name(
        product
    )

    f = product_info_file(
        product
    )

    info = default_product_info(
        product
    )

    if f.exists():

        try:

            data = json.loads(
                f.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(
                data,
                dict
            ):

                info.update(
                    {
                        k:
                        str(
                            data.get(
                                k,
                                ""
                            ) or ""
                        )

                        for k in info

                        if k != "product"
                    }
                )

        except Exception:

            pass

    return info


def save_product_info(
    product: str,
    info: dict
):

    product = validate_product_name(
        product
    )

    target = product_info_file(
        product
    )

    data = default_product_info(
        product
    )

    for key in data:

        if key != "product":

            data[key] = str(
                info.get(
                    key,
                    ""
                ) or ""
            ).strip()

    save_script_atomic(
        target,
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        )
    )


def create_product(
    product: str,
    script: str = "",
    info: dict | None = None
):

    product = validate_product_name(
        product
    )

    script_dir = (
        SCRIPT_DIR /
        product
    )

    output_dir = (
        OUTPUT_DIR /
        product
    )

    if (
        script_dir.exists()
        or
        output_dir.exists()
    ):

        raise RuntimeError(
            f"商品「{product}」已经存在。"
        )

    script_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    try:

        content = (
            script or ""
        ).strip()

        if not content:

            content = default_product_script(
                product
            )

        if not tts.split_script(
            content
        ):

            raise RuntimeError(
                "初始话术没有识别到有效的 [标签] 段落。"
            )

        (
            script_dir /
            "script.txt"
        ).write_text(
            content + "\n",
            encoding="utf-8"
        )

        save_product_info(
            product,
            info or {}
        )

        return product

    except Exception:

        import shutil

        shutil.rmtree(
            script_dir,
            ignore_errors=True
        )

        raise


def delete_product(product: str):

    import shutil

    product = validate_product_name(
        product
    )

    if product not in get_products():

        raise RuntimeError(
            f"商品「{product}」不存在。"
        )

    current = get_state()

    if (
        current["product"] == product
        and
        current["status"]
        in (
            "starting",
            "playing",
            "pause_requested",
            "paused"
        )
    ):

        raise RuntimeError(
            "该商品正在直播，请先停止直播后再删除。"
        )

    if (
        get_tts_state().get(
            "product"
        ) == product
        and
        get_tts_state().get(
            "status"
        )
        in (
            "loading",
            "generating"
        )
    ):

        raise RuntimeError(
            "该商品正在生成 WAV，请等待完成后再删除。"
        )

    if (
        get_ai_state().get(
            "product"
        ) == product
        and
        get_ai_state().get(
            "status"
        ) == "generating"
    ):

        raise RuntimeError(
            "该商品正在生成 AI 话术，请等待完成后再删除。"
        )

    shutil.rmtree(
        SCRIPT_DIR / product,
        ignore_errors=True
    )

    shutil.rmtree(
        OUTPUT_DIR / product,
        ignore_errors=True
    )

    return True


def get_products():

    if not SCRIPT_DIR.exists():

        return []

    products = []

    for p in SCRIPT_DIR.iterdir():

        if (
            p.is_dir()
            and
            not p.name.startswith(".")
        ):

            products.append(
                p.name
            )

    return sorted(products)


# =========================================================
# 读取商品话术
# =========================================================

def read_product_script(product: str):

    script_file = (
        SCRIPT_DIR /
        product /
        "script.txt"
    )

    if not script_file.exists():

        raise FileNotFoundError(
            f"找不到话术文件：{script_file}"
        )

    return script_file.read_text(
        encoding="utf-8"
    )


# =========================================================
# WAV
# =========================================================

def get_audio_items(
    product: str,
    style: str | None = None
):

    if style:

        script_file = script_version_file(
            product,
            style
        )

        version_key = SCRIPT_STYLES.get(
            style,
            ""
        )

        output_dir = (
            OUTPUT_DIR /
            product /
            version_key
        )

    else:

        script_file = (
            SCRIPT_DIR /
            product /
            "script.txt"
        )

        output_dir = (
            OUTPUT_DIR /
            product
        )

    if not script_file.exists():

        return []

    script = script_file.read_text(
        encoding="utf-8"
    )

    chunks = tts.split_script(
        script
    )

    if (
        not chunks
        or
        not output_dir.exists()
    ):

        return []

    audio_items = []

    for i, chunk in enumerate(
        chunks
    ):

        if isinstance(
            chunk,
            dict
        ):

            tag = chunk.get(
                "tag",
                chunk.get(
                    "title",
                    f"第{i + 1}段"
                )
            )

            text = chunk.get(
                "text",
                ""
            )

        elif isinstance(
            chunk,
            (tuple, list)
        ):

            tag = (
                str(chunk[0])
                if chunk
                else
                f"第{i + 1}段"
            )

            text = (
                str(chunk[1])
                if len(chunk) > 1
                else
                ""
            )

        else:

            tag = f"第{i + 1}段"

            text = str(chunk)

        safe_tag = tts.safe_filename(
            tag
        )

        candidates = [

            output_dir /
            f"{i + 1:03d}_{safe_tag}.wav",

            output_dir /
            f"{i + 1:02d}_{safe_tag}.wav",

            output_dir /
            f"{i + 1:02d}.wav",

            output_dir /
            f"{i + 1:03d}.wav",

            output_dir /
            f"segment_{i + 1:02d}.wav",

            output_dir /
            f"segment_{i + 1:03d}.wav",

        ]

        wav_file = next(
            (
                x
                for x in candidates
                if x.exists()
            ),
            None
        )

        if not wav_file:

            return []

        voice_style = tts.VOICE_STYLE.get(
            tag,
            tts.DEFAULT_STYLE
        )

        audio_items.append(
            {

                "wav_file":
                    str(wav_file),

                "title":
                    tag,

                "text":
                    text,

                "pause":
                    voice_style[
                        "pause"
                    ],

            }
        )

    return audio_items


def get_playback_versions(product: str):

    versions = []

    for style in SCRIPT_STYLES:

        items = get_audio_items(
            product,
            style
        )

        if items:

            versions.append(
                {
                    "style":
                        style,

                    "items":
                        items
                }
            )

    if not versions:

        items = get_audio_items(
            product
        )

        if items:

            versions.append(
                {
                    "style":
                        "当前话术",

                    "items":
                        items
                }
            )

    return versions


# =========================================================
# 停止声卡
# =========================================================

def stop_audio_device():

    try:

        tts.sd.stop()

    except Exception:

        pass


# =========================================================
# 播放线程
# =========================================================

def playback_worker(
    product: str,
    playback_versions
):

    global play_thread

    try:

        if not playback_versions:

            raise RuntimeError(
                "没有可播放的完整 WAV 版本，请先生成 WAV。"
            )

        round_no = 0

        last_style = None

        set_state(

            status="playing",

            mode="live_loop",

            product=product,

            current_index=0,

            total=0,

            current_title="",

            current_text="",

            current_version="",

            current_audio="",

            current_audio_duration=0,

            round_no=0,

            progress_percent=0,

            alert_level="ok",

            alert_message="",

            message="直播播放中",

        )

        add_live_log(
            f"直播启动：商品「{product}」"
        )

        while not stop_event.is_set():

            choices = [

                v

                for v in playback_versions

                if
                len(playback_versions) == 1
                or
                v["style"] != last_style

            ]

            selected = random.choice(
                choices
                or
                playback_versions
            )

            last_style = selected[
                "style"
            ]

            items = selected[
                "items"
            ]

            round_no += 1

            total = len(items)

            set_state(

                status="playing",

                product=product,

                current_index=0,

                total=total,

                current_title="",

                current_text="",

                current_audio="",

                current_audio_duration=0,

                current_version=last_style,

                round_no=round_no,

                progress_percent=0,

                alert_level="ok",

                alert_message="",

                message=
                    f"第 {round_no} 轮：随机选择「{last_style}」",

            )

            add_live_log(
                f"第 {round_no} 轮开始：随机选择「{last_style}」，共 {total} 段"
            )

            index = 0

            while (
                index < total
                and
                not stop_event.is_set()
            ):

                if pause_event.is_set():

                    set_state(

                        status="paused",

                        message="已暂停，等待继续播放"

                    )

                    add_live_log(
                        "直播已暂停，等待继续播放"
                    )

                    while (
                        pause_event.is_set()
                        and
                        not stop_event.is_set()
                    ):

                        time.sleep(
                            0.1
                        )

                    if stop_event.is_set():

                        break

                    set_state(

                        status="playing",

                        message=
                            f"第 {round_no} 轮继续播放「{last_style}」"

                    )

                    add_live_log(
                        "直播继续播放"
                    )

                item = items[index]

                percent = (
                    int(
                        (index + 1)
                        * 100
                        / total
                    )
                    if total
                    else
                    0
                )

                wav_file = item[
                    "wav_file"
                ]

                audio_name = Path(
                    wav_file
                ).name

                audio_duration = get_wav_duration(
                    wav_file
                )

                set_state(

                    status="playing",

                    current_index=index + 1,

                    total=total,

                    current_title=item[
                        "title"
                    ],

                    current_text=item.get(
                        "text",
                        ""
                    ),

                    current_version=
                        last_style,

                    current_audio=
                        audio_name,

                    current_audio_duration=
                        audio_duration,

                    round_no=
                        round_no,

                    progress_percent=
                        percent,

                    alert_level="ok",

                    alert_message="",

                    message=
                        f"第 {round_no} 轮 · {last_style} · 第 {index + 1} / {total} 段",

                )

                add_live_log(
                    f"播放第 {index + 1}/{total} 段：{item['title']} · {audio_name}"
                )

                try:

                    tts.play_wav(
                        wav_file,
                        1.0
                    )

                except Exception as audio_error:

                    set_state(

                        alert_level="error",

                        alert_message=
                            f"音频播放失败：{audio_error}",

                        message=
                            f"播放出错：{audio_error}"

                    )

                    add_live_log(
                        f"❌ 音频播放失败：{audio_error}"
                    )

                    raise

                if stop_event.is_set():

                    break

                if pause_event.is_set():

                    set_state(

                        status="paused",

                        message=
                            "本句话已播放完，已暂停"

                    )

                    add_live_log(
                        "当前话术播放完成，已暂停"
                    )

                    while (
                        pause_event.is_set()
                        and
                        not stop_event.is_set()
                    ):

                        time.sleep(
                            0.1
                        )

                    if stop_event.is_set():

                        break

                    set_state(

                        status="playing",

                        message=
                            f"第 {round_no} 轮继续播放「{last_style}」"

                    )

                    add_live_log(
                        "直播继续播放"
                    )

                pause_time = float(
                    item.get(
                        "pause",
                        0.7
                    )
                )

                if stop_event.wait(
                    pause_time
                ):

                    break

                index += 1

        set_state(

            status="stopped",

            current_index=0,

            current_title="",

            current_text="",

            current_version="",

            current_audio="",

            current_audio_duration=0,

            progress_percent=0,

            round_no=0,

            live_stopped_at=
                player_state.get(
                    "live_stopped_at"
                )
                or
                time.time(),

            message="直播已停止",

        )

        add_live_log(
            "直播已停止"
        )

    except Exception as e:

        set_state(

            status="stopped",

            alert_level="error",

            alert_message=str(e),

            message=
                f"播放出错：{e}",

            live_stopped_at=
                time.time(),

        )

        add_live_log(
            f"❌ 播放出错：{e}"
        )

    finally:

        stop_audio_device()

        play_thread = None


# =========================================================
# 开始播放
# =========================================================

def start_playback(
    product: str
):

    global play_thread

    product = (
        product or ""
    ).strip()

    if product not in get_products():

        raise RuntimeError(
            f"商品「{product}」不存在。"
        )

    playback_versions = get_playback_versions(
        product
    )

    if not playback_versions:

        raise RuntimeError(
            f"商品「{product}」没有完整 WAV，请先生成全部 WAV。"
        )

    with play_lock:

        if (
            play_thread
            and
            play_thread.is_alive()
        ):

            current = get_state()

            if current[
                "status"
            ] in (
                "paused",
                "pause_requested"
            ):

                raise RuntimeError(
                    "当前直播已经暂停，请点击「继续播放」。"
                )

            raise RuntimeError(
                "当前已经有直播正在播放，请先停止当前直播。"
            )

        stop_event.clear()

        pause_event.clear()

        with state_lock:

            player_state[
                "logs"
            ] = []

        now = time.time()

        set_state(

            status="starting",

            product=product,

            current_index=0,

            total=0,

            current_title="",

            current_text="",

            current_version="",

            current_audio="",

            current_audio_duration=0,

            round_no=0,

            progress_percent=0,

            live_started_at=now,

            live_stopped_at=None,

            alert_level="ok",

            alert_message="",

            message=
                f"正在启动「{product}」直播循环..."

        )

        add_live_log(
            f"正在启动商品「{product}」直播"
        )

        play_thread = threading.Thread(

            target=playback_worker,

            args=(
                product,
                playback_versions
            ),

            daemon=True,

        )

        play_thread.start()


# =========================================================
# 暂停
# =========================================================

def request_pause():

    current = get_state()

    if current[
        "status"
    ] != "playing":

        return current

    pause_event.set()

    set_state(

        status="pause_requested",

        message=
            "已请求暂停，当前这句话播放完后暂停"

    )

    add_live_log(
        "收到暂停指令：当前一句播放完成后暂停"
    )

    return get_state()


# =========================================================
# 继续
# =========================================================

def resume_playback():

    current = get_state()

    if current[
        "status"
    ] not in (
        "paused",
        "pause_requested"
    ):

        return current

    pause_event.clear()

    set_state(

        status="playing",

        message="继续播放"

    )

    add_live_log(
        "收到继续播放指令"
    )

    return get_state()


# =========================================================
# 停止
# =========================================================

def stop_playback():

    stop_event.set()

    pause_event.clear()

    stop_audio_device()

    now = time.time()

    set_state(

        status="stopped",

        current_index=0,

        current_title="",

        current_text="",

        current_version="",

        current_audio="",

        current_audio_duration=0,

        progress_percent=0,

        live_stopped_at=now,

        message="直播已停止",

    )

    add_live_log(
        "收到停止直播指令"
    )


# =========================================================
# CosyVoice 加载
# =========================================================

def load_cosyvoice():

    global cosyvoice

    if cosyvoice is not None:

        return cosyvoice

    with model_lock:

        if cosyvoice is not None:

            return cosyvoice

        print()

        print(
            "=" * 60
        )

        print(
            "正在加载 CosyVoice3..."
        )

        print(
            "=" * 60
        )

        start = time.time()

        cosyvoice = tts.AutoModel(

            model_dir=
                tts.MODEL_DIR

        )

        elapsed = (
            time.time()
            - start
        )

        print(
            f"✓ CosyVoice3 加载完成，耗时 {elapsed:.2f} 秒"
        )

    return cosyvoice


# =========================================================
# 后台生成全部 WAV
# =========================================================

def tts_generation_worker(
    product: str
):

    global tts_thread

    try:

        set_tts_state(

            status="loading",

            product=product,

            current_index=0,

            total=0,

            percent=0,

            current_tag="",

            message=
                "正在准备 CosyVoice...",

            error=""

        )

        model = load_cosyvoice()

        version_sources = []

        for style in SCRIPT_STYLES:

            f = script_version_file(
                product,
                style
            )

            if f.exists():

                version_sources.append(
                    (
                        style,
                        f
                    )
                )

        if not version_sources:

            version_sources = [
                (
                    "当前话术",
                    SCRIPT_DIR /
                    product /
                    "script.txt"
                )
            ]

        parsed = []

        total = 0

        for style, script_file in version_sources:

            script = script_file.read_text(
                encoding="utf-8"
            ).strip()

            chunks = tts.split_script(
                script
            )

            if chunks:

                parsed.append(
                    (
                        style,
                        chunks
                    )
                )

                total += len(
                    chunks
                )

        if not parsed or total == 0:

            raise RuntimeError(
                "当前商品没有识别到有效话术。"
            )

        product_output_root = (
            OUTPUT_DIR /
            product
        )

        product_output_root.mkdir(
            parents=True,
            exist_ok=True
        )

        clear_product_audio(
            product
        )

        product_output_root = (
            OUTPUT_DIR /
            product
        )

        for style, _ in parsed:

            key = SCRIPT_STYLES.get(
                style
            )

            if key:

                (
                    product_output_root /
                    key
                ).mkdir(
                    parents=True,
                    exist_ok=True
                )

        completed = 0

        audio_summary = []

        for style, chunks in parsed:

            key = SCRIPT_STYLES.get(
                style
            )

            output_dir = (
                product_output_root /
                key
                if key
                else
                product_output_root
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            version_items = []

            for i, chunk in enumerate(
                chunks,
                start=1
            ):

                tag = chunk[
                    "tag"
                ]

                text = chunk[
                    "text"
                ]

                voice_style = tts.VOICE_STYLE.get(
                    tag,
                    tts.DEFAULT_STYLE
                )

                safe_tag = tts.safe_filename(
                    tag
                )

                output_file = (
                    output_dir /
                    f"{i:03d}_{safe_tag}.wav"
                )

                set_tts_state(

                    status="generating",

                    product=product,

                    current_index=
                        completed + 1,

                    total=total,

                    percent=
                        int(
                            completed
                            * 100
                            / total
                        ),

                    current_tag=tag,

                    current_version=style,

                    message=
                        f"{style}：正在生成第 {i} / {len(chunks)} 段（总进度 {completed + 1} / {total}）",

                )

                result = tts.generate_tts(

                    model,

                    text,

                    str(output_file)

                )

                if result is None:

                    raise RuntimeError(
                        f"「{style}」第 {i} 段「{tag}」生成失败。"
                    )

                version_items.append(

                    {
                        "index":
                            i,

                        "tag":
                            tag,

                        "wav_file":
                            str(output_file),

                        "audio_duration":
                            result.get(
                                "audio_duration",
                                0
                            )
                    }

                )

                completed += 1

                set_tts_state(

                    status="generating",

                    product=product,

                    current_index=
                        completed,

                    total=total,

                    percent=
                        int(
                            completed
                            * 100
                            / total
                        ),

                    current_tag=tag,

                    current_version=style,

                    message=
                        f"{style}：已完成第 {i} / {len(chunks)} 段（总进度 {completed} / {total}）",

                )

            audio_summary.append(
                (
                    style,
                    version_items
                )
            )

            playlist_file = (
                output_dir /
                "playlist.txt"
            )

            with playlist_file.open(
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"商品：{product}\n"
                )

                f.write(
                    f"话术版本：{style}\n"
                )

                f.write(
                    f"总段数：{len(chunks)}\n\n"
                )

                for item in version_items:

                    f.write(

                        f"{item['index']:03d} | "
                        f"{item['tag']} | "
                        f"{item['wav_file']}\n"

                    )

        set_tts_state(

            status="completed",

            product=product,

            current_index=total,

            total=total,

            percent=100,

            current_tag="",

            current_version="",

            message=
                f"全部 WAV 生成完成，共 {len(parsed)} 套话术、{total} 段。直播时每轮随机选择。",

            error="",

        )

        print(
            f"✓ 商品「{product}」全部 WAV 生成完成，共 {len(parsed)} 套话术"
        )

    except Exception as e:

        print(
            f"TTS 生成任务出错：{e}"
        )

        set_tts_state(

            status="error",

            message="生成失败",

            error=str(e)

        )

    finally:

        tts_thread = None


# =========================================================
# 启动后台生成
# =========================================================

def start_tts_generation(
    product: str
):

    global tts_thread

    with tts_lock:

        if (
            tts_thread
            and
            tts_thread.is_alive()
        ):

            current = get_tts_state()

            raise RuntimeError(

                f"当前正在生成「{current.get('product', '')}」，请等待完成。"

            )

        tts_thread = threading.Thread(

            target=tts_generation_worker,

            args=(product,),

            daemon=True,

        )

        set_tts_state(

            status="loading",

            product=product,

            current_index=0,

            total=0,

            percent=0,

            current_tag="",

            current_version="",

            message="任务已启动",

            error=""

        )

        tts_thread.start()


# =========================================================
# DeepSeek AI 话术生成
# =========================================================

def generate_script_with_deepseek(
    product: str,
    old_script: str,
    style: str = "高转化直播型"
) -> str:

    api_key = os.getenv(
        "DEEPSEEK_API_KEY",
        ""
    ).strip()

    if not api_key:

        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY，请先设置 DeepSeek API Key。"
        )

    style_rules = {

        "高转化直播型":
            "重点强化开场吸引力、卖点节奏、购买理由和自然促单，但不能夸大宣传。",

        "自然聊天型":
            "像主播和观众聊天一样自然，减少销售腔，多用短句和生活化表达，让观众听起来舒服。",

        "强种草型":
            "重点突出使用场景、实际体验和商品卖点，让观众产生兴趣，但不能虚构体验或效果。",
    }

    style_instruction = style_rules.get(
        style,
        style_rules[
            "高转化直播型"
        ]
    )

    product_info = get_product_info(
        product
    )

    system_prompt = f"""
你是一名经验丰富的中文直播带货主播兼话术策划师。
现在需要把现有商品话术重新整理成一版可以直接用于真人/AI直播的口播稿。

【本次话术风格】
{style}
{style_instruction}

【最重要的规则】
1. 必须完整保留原稿所有 [标签]，标签名称、数量、顺序必须完全一致。
2. 每个 [标签] 下都必须有内容，不能删除标签，也不能新增标签。
3. 商品事实只能来自原稿。原稿没有明确写出的品牌、型号、规格、材质、成分、价格、优惠、库存、销量、产地、资质、功效、认证等，一律不得自行补充。
4. 不得虚构“全网最低”“全网第一”“唯一”“绝对有效”“100%”“国家级”等无法由原稿证明的内容。
5. 不得把推测、常识或营销经验写成商品事实。
6. 可以重写表达方式，但不能改变原稿中的商品事实、数字、单位和关键条件。
7. 语言必须适合语音播报：以短句为主，一句话尽量表达一个意思；少用复杂长句、书面语和生僻词。
8. 要有直播现场感，可以自然加入“大家可以看一下”“喜欢的朋友可以了解一下”等口语，但不要机械重复。
9. 不要写标题、说明、分析、修改理由，不要使用 Markdown 代码块。
10. 只输出最终完整 script.txt 内容。

【商品资料】
以下资料是本次创作允许使用的商品事实来源。资料为空的字段不得自行补充：
- 商品名称：{product_info.get("product", product)}
- 价格：{product_info.get("price", "")}
- 规格：{product_info.get("specification", "")}
- 核心卖点：{product_info.get("selling_points", "")}
- 适用人群：{product_info.get("target_people", "")}
- 使用场景：{product_info.get("usage_scene", "")}
- 注意事项：{product_info.get("notes", "")}
- 其他补充：{product_info.get("extra", "")}

【口播要求】
- 开场要尽快进入商品，不要长篇寒暄。
- 卖点之间要有自然过渡，不要像产品说明书逐条念参数。
- 促单要自然，不得制造虚假紧迫感。
- 不要频繁使用“姐妹们”“家人们”“宝子们”等网络称呼。
- 不要连续重复同一个句式。
- 保留原稿中的数字、价格、规格、时间等信息，不得擅自改数字。
""".strip()

    user_prompt = f"""
商品名称：{product}
话术风格：{style}

商品资料：
价格：{product_info.get("price", "")}
规格：{product_info.get("specification", "")}
核心卖点：{product_info.get("selling_points", "")}
适用人群：{product_info.get("target_people", "")}
使用场景：{product_info.get("usage_scene", "")}
注意事项：{product_info.get("notes", "")}
其他补充：{product_info.get("extra", "")}

下面是当前正在使用的 script.txt。
请严格以它为事实来源，在保留全部 [标签] 结构的前提下重新创作。

--- 原话术开始 ---
{old_script}
--- 原话术结束 ---

请直接输出新的完整 script.txt，不要输出任何解释。
""".strip()

    headers = {

        "Authorization":
            f"Bearer {api_key}",

        "Content-Type":
            "application/json",

    }

    payload = {

        "model":
            DEEPSEEK_MODEL,

        "messages": [

            {
                "role":
                    "system",

                "content":
                    system_prompt
            },

            {
                "role":
                    "user",

                "content":
                    user_prompt
            },

        ],

        "temperature":
            0.75,

        "stream":
            False,

    }

    try:

        response = requests.post(

            DEEPSEEK_API_URL,

            headers=headers,

            json=payload,

            timeout=180,

        )

    except requests.RequestException as e:

        raise RuntimeError(
            f"无法连接 DeepSeek API：{e}"
        )

    try:

        data = response.json()

    except Exception:

        data = {}

    if not response.ok:

        detail = (
            data.get(
                "error",
                {}
            )
            if isinstance(
                data,
                dict
            )
            else
            {}
        )

        if isinstance(
            detail,
            dict
        ):

            detail = (
                detail.get(
                    "message"
                )
                or
                detail.get(
                    "type"
                )
                or
                str(detail)
            )

        raise RuntimeError(

            f"DeepSeek API 请求失败（HTTP {response.status_code}）："
            f"{detail or response.text[:300]}"

        )

    try:

        content = data[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

    except (
        KeyError,
        IndexError,
        TypeError
    ):

        raise RuntimeError(
            "DeepSeek 返回结果格式异常，未找到生成的话术内容。"
        )

    content = str(
        content
    ).strip()

    if not content:

        raise RuntimeError(
            "DeepSeek 返回了空话术。"
        )

    if (
        content.startswith("```")
        and
        content.endswith("```")
    ):

        lines = content.splitlines()

        if len(lines) >= 3:

            content = "\n".join(
                lines[1:-1]
            ).strip()

    return content


# =========================================================
# 话术版本
# =========================================================

SCRIPT_STYLES = {

    "高转化直播型":
        "high_convert",

    "自然聊天型":
        "natural",

    "强种草型":
        "seed",
}


STYLE_KEYS = {
    v: k
    for k, v in SCRIPT_STYLES.items()
}


def script_version_file(
    product: str,
    style: str
) -> Path:

    key = SCRIPT_STYLES.get(
        style
    )

    if not key:

        raise RuntimeError(
            f"不支持的话术风格：{style}"
        )

    return (
        SCRIPT_DIR /
        product /
        f"script_{key}.txt"
    )


def validate_script(
    old_script: str,
    new_script: str
):

    old_tags = [
        x.get(
            "tag",
            ""
        )
        for x in tts.split_script(
            old_script
        )
    ]

    new_tags = [
        x.get(
            "tag",
            ""
        )
        for x in tts.split_script(
            new_script
        )
    ]

    if old_tags != new_tags:

        raise RuntimeError(
            "AI 生成结果的 [标签] 与原话术不一致，为安全起见未保存。"
        )


def save_script_atomic(
    path: Path,
    content: str
):

    temp_file = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp_file.write_text(
        content.strip() + "\n",
        encoding="utf-8"
    )

    temp_file.replace(
        path
    )


def clear_product_audio(
    product: str
):

    import shutil

    product_output_dir = (
        OUTPUT_DIR /
        product
    )

    if product_output_dir.exists():

        shutil.rmtree(
            product_output_dir,
            ignore_errors=True
        )

    product_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# AI 话术线程
# =========================================================

def ai_script_worker(
    product: str,
    style: str = "高转化直播型"
):

    global ai_thread

    try:

        set_ai_state(

            status="generating",

            product=product,

            current=1,

            total=1,

            percent=0,

            current_style=style,

            message=
                f"正在调用 DeepSeek 生成「{style}」...",

            error="",

        )

        script_file = (
            SCRIPT_DIR /
            product /
            "script.txt"
        )

        if not script_file.exists():

            raise RuntimeError(
                "找不到当前商品的 script.txt。"
            )

        old_script = script_file.read_text(
            encoding="utf-8"
        ).strip()

        if not old_script:

            raise RuntimeError(
                "当前 script.txt 为空，无法重新生成话术。"
            )

        new_script = generate_script_with_deepseek(

            product,

            old_script,

            style

        )

        validate_script(
            old_script,
            new_script
        )

        save_script_atomic(

            script_version_file(
                product,
                style
            ),

            new_script

        )

        save_script_atomic(

            script_file,

            new_script

        )

        clear_product_audio(
            product
        )

        set_ai_state(

            status="completed",

            product=product,

            current=1,

            total=1,

            percent=100,

            current_style=style,

            message=
                f"「{style}」生成完成，已设为当前版本，旧 WAV 已清理。",

            error="",

        )

        print(
            f"✓ 商品「{product}」DeepSeek {style} 话术生成完成"
        )

    except Exception as e:

        print(
            f"AI 话术生成任务出错：{e}"
        )

        set_ai_state(

            status="error",

            product=product,

            message="AI话术生成失败",

            error=str(e)

        )

    finally:

        ai_thread = None


def ai_all_styles_worker(
    product: str
):

    global ai_thread

    try:

        script_file = (
            SCRIPT_DIR /
            product /
            "script.txt"
        )

        if not script_file.exists():

            raise RuntimeError(
                "找不到当前商品的 script.txt。"
            )

        old_script = script_file.read_text(
            encoding="utf-8"
        ).strip()

        if not old_script:

            raise RuntimeError(
                "当前 script.txt 为空，无法生成话术。"
            )

        styles = list(
            SCRIPT_STYLES.keys()
        )

        for index, style in enumerate(
            styles,
            1
        ):

            set_ai_state(

                status="generating",

                product=product,

                current=index,

                total=len(styles),

                percent=
                    int(
                        (index - 1)
                        * 100
                        / len(styles)
                    ),

                current_style=style,

                message=
                    f"正在生成第 {index} / {len(styles)} 套：{style}...",

                error=""

            )

            new_script = generate_script_with_deepseek(

                product,

                old_script,

                style

            )

            validate_script(
                old_script,
                new_script
            )

            save_script_atomic(

                script_version_file(
                    product,
                    style
                ),

                new_script

            )

        active = script_version_file(

            product,

            styles[0]

        )

        save_script_atomic(

            script_file,

            active.read_text(
                encoding="utf-8"
            )

        )

        clear_product_audio(
            product
        )

        set_ai_state(

            status="completed",

            product=product,

            current=len(styles),

            total=len(styles),

            percent=100,

            current_style=styles[0],

            message=
                "3套 AI 话术全部生成完成，当前使用「高转化直播型」，旧 WAV 已清理。",

            error=""

        )

        print(
            f"✓ 商品「{product}」3套 DeepSeek 话术生成完成"
        )

    except Exception as e:

        print(
            f"AI 多版本生成任务出错：{e}"
        )

        set_ai_state(

            status="error",

            product=product,

            message="AI多版本话术生成失败",

            error=str(e)

        )

    finally:

        ai_thread = None


def start_ai_script_generation(

    product: str,

    style: str =
        "高转化直播型",

    all_styles: bool =
        False

):

    global ai_thread

    with ai_lock:

        if (
            ai_thread
            and
            ai_thread.is_alive()
        ):

            current = get_ai_state()

            raise RuntimeError(

                f"当前正在生成「{current.get('product', '')}」的话术，请等待完成。"

            )

        target = (
            ai_all_styles_worker
            if all_styles
            else
            ai_script_worker
        )

        args = (
            (product,)
            if all_styles
            else
            (product, style)
        )

        ai_thread = threading.Thread(

            target=target,

            args=args,

            daemon=True,

        )

        set_ai_state(

            status="generating",

            product=product,

            current=0,

            total=
                3
                if all_styles
                else
                1,

            percent=0,

            current_style="",

            message="任务已启动...",

            error=""

        )

        ai_thread.start()


def get_script_versions(
    product: str
):

    versions = []

    active = (
        SCRIPT_DIR /
        product /
        "script.txt"
    )

    active_text = (
        active.read_text(
            encoding="utf-8"
        )
        if active.exists()
        else
        ""
    )

    for style, key in SCRIPT_STYLES.items():

        f = (
            SCRIPT_DIR /
            product /
            f"script_{key}.txt"
        )

        if f.exists():

            versions.append(

                {

                    "style":
                        style,

                    "key":
                        key,

                    "exists":
                        True,

                    "active":
                        f.read_text(
                            encoding="utf-8"
                        )
                        == active_text,

                }

            )

        else:

            versions.append(

                {

                    "style":
                        style,

                    "key":
                        key,

                    "exists":
                        False,

                    "active":
                        False,

                }

            )

    return versions


def select_script_version(
    product: str,
    style: str
):

    source = script_version_file(
        product,
        style
    )

    if not source.exists():

        raise RuntimeError(
            f"「{style}」还没有生成，请先生成该版本。"
        )

    target = (
        SCRIPT_DIR /
        product /
        "script.txt"
    )

    save_script_atomic(

        target,

        source.read_text(
            encoding="utf-8"
        )

    )

    clear_product_audio(
        product
    )


# =========================================================
# 网页 HTML
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>AI直播控制台 V8</title>

<style>

/* =========================================================
   全局
   ========================================================= */

* {
    box-sizing: border-box;
}

html,
body {
    width: 100%;
    height: 100%;
}

body {

    margin: 0;

    font-family:
        "Microsoft YaHei",
        "Segoe UI",
        Arial,
        sans-serif;

    background:
        #080b12;

    color:
        #e8edf5;

    overflow: hidden;
}


/* =========================================================
   主容器
   ========================================================= */

.app {

    width: 100%;

    height: 100vh;

    display: flex;

    flex-direction: column;

}


/* =========================================================
   顶部
   ========================================================= */

.topbar {

    height: 68px;

    flex-shrink: 0;

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding:
        0 24px;

    background:
        #0d111a;

    border-bottom:
        1px solid #202735;

}

.brand {

    display: flex;

    align-items: center;

    gap: 14px;

}

.brand-icon {

    width: 42px;

    height: 42px;

    border-radius: 12px;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 23px;

    background:
        #171d29;

    border:
        1px solid #293242;

}

.brand-title {

    font-size: 21px;

    font-weight: 700;

    letter-spacing: .3px;

}

.brand-subtitle {

    margin-top: 2px;

    font-size: 12px;

    color:
        #7f8ba0;

}

.clock {

    color:
        #8e9bb0;

    font-size: 13px;

}


/* =========================================================
   主体两栏
   ========================================================= */

.main {

    flex: 1;

    min-height: 0;

    display: grid;

    grid-template-columns:
        minmax(390px, 42%)
        minmax(0, 58%);

    gap: 0;

}


/* =========================================================
   左侧控制区
   ========================================================= */

.left {

    min-width: 0;

    min-height: 0;

    overflow-y: auto;

    padding: 18px;

    background:
        #10151f;

    border-right:
        1px solid #202735;

}

.left::-webkit-scrollbar,
.monitor::-webkit-scrollbar,
.script-box::-webkit-scrollbar,
.log-box::-webkit-scrollbar {

    width: 7px;

}

.left::-webkit-scrollbar-thumb,
.monitor::-webkit-scrollbar-thumb,
.script-box::-webkit-scrollbar-thumb,
.log-box::-webkit-scrollbar-thumb {

    background:
        #303949;

    border-radius: 8px;

}


/* =========================================================
   右侧监控
   ========================================================= */

.monitor {

    min-width: 0;

    min-height: 0;

    overflow-y: auto;

    padding: 18px;

    background:
        #080b12;

}


/* =========================================================
   面板
   ========================================================= */

.panel {

    background:
        #151b26;

    border:
        1px solid #252e3d;

    border-radius: 14px;

    padding: 16px;

    margin-bottom: 14px;

    box-shadow:
        0 8px 24px rgba(0,0,0,.16);

}

.panel-title {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 10px;

    margin-bottom: 13px;

}

.panel-title h2 {

    margin: 0;

    font-size: 16px;

    font-weight: 700;

}

.panel-title .small {

    color:
        #778398;

    font-size: 12px;

}


/* =========================================================
   直播控制 —— 左上核心区域
   ========================================================= */

.live-control {

    border:
        1px solid #31415a;

    background:
        #121b2a;

    box-shadow:
        0 10px 35px rgba(0,0,0,.28);

}

.live-control-head {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 14px;

}

.live-control-title {

    font-size: 18px;

    font-weight: 700;

}

.live-control-status {

    display: flex;

    align-items: center;

    gap: 7px;

    font-size: 12px;

    color:
        #91a0b5;

}

.status-dot {

    width: 9px;

    height: 9px;

    border-radius: 50%;

    background:
        #667085;

}

.status-dot.playing {

    background:
        #35d07f;

    box-shadow:
        0 0 0 5px rgba(53,208,127,.12),
        0 0 15px rgba(53,208,127,.6);

    animation:
        pulse 1.5s infinite;

}

.status-dot.paused,
.status-dot.starting {

    background:
        #f5b83d;

    box-shadow:
        0 0 12px rgba(245,184,61,.4);

}

.status-dot.error {

    background:
        #ff5c68;

    box-shadow:
        0 0 12px rgba(255,92,104,.5);

}

@keyframes pulse {

    0%,100% {
        opacity: 1;
    }

    50% {
        opacity: .45;
    }

}


/* =========================================================
   控制按钮
   ========================================================= */

.live-buttons {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 10px;

}

button {

    border: 0;

    cursor: pointer;

    font-family: inherit;

    transition:
        transform .12s ease,
        filter .12s ease,
        opacity .12s ease;

}

button:not(:disabled):hover {

    filter:
        brightness(1.08);

}

button:not(:disabled):active {

    transform:
        scale(.98);

}

button:disabled {

    opacity: .35;

    cursor:
        not-allowed;

}

.live-btn {

    min-height: 55px;

    border-radius: 11px;

    font-size: 15px;

    font-weight: 700;

}

.btn-start {

    color: #fff;

    background:
        #1677ff;

}

.btn-pause {

    color: #fff;

    background:
        #c88b17;

}

.btn-resume {

    color: #fff;

    background:
        #2da45a;

}

.btn-stop {

    color: #fff;

    background:
        #d93b48;

}


/* =========================================================
   直播控制提示
   ========================================================= */

.live-hint {

    margin-top: 12px;

    padding:
        10px 12px;

    border-radius: 9px;

    background:
        #0c111a;

    color:
        #8390a4;

    font-size: 12px;

    line-height: 1.7;

}


/* =========================================================
   商品
   ========================================================= */

select,
input,
textarea {

    font-family: inherit;

}

select {

    width: 100%;

    min-width: 0;

    padding:
        11px 12px;

    border:
        1px solid #303a4b;

    border-radius: 9px;

    background:
        #0d121b;

    color:
        #e7edf5;

    outline: none;

}

select:focus,
input:focus,
textarea:focus {

    border-color:
        #377dff;

}

.row {

    display: flex;

    gap: 8px;

    flex-wrap: wrap;

    align-items: center;

}

.tool-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 8px;

    margin-top: 9px;

}

.tool-btn {

    padding:
        10px;

    border-radius: 9px;

    background:
        #202735;

    color:
        #dce3ed;

    border:
        1px solid #303a4b;

    font-size: 13px;

}

.btn-purple {

    background:
        #633bb8;

    color: white;

}

.btn-cyan {

    background:
        #148f98;

    color: white;

}


/* =========================================================
   AI
   ========================================================= */

.ai-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 8px;

}

.ai-grid button {

    padding:
        10px;

    border-radius: 9px;

    font-size: 13px;

}

.ai-message {

    margin-top: 9px;

    padding:
        9px 11px;

    border-radius: 8px;

    background:
        #0d121b;

    color:
        #8c99ad;

    font-size: 12px;

    line-height: 1.6;

}


/* =========================================================
   输入
   ========================================================= */

.info-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 8px;

}

.info-grid input,
.panel textarea {

    width: 100%;

    padding:
        10px 11px;

    border:
        1px solid #303a4b;

    border-radius: 9px;

    background:
        #0d121b;

    color:
        #e7edf5;

    outline: none;

}

.panel textarea {

    margin-top: 8px;

    resize: vertical;

    line-height: 1.6;

}


/* =========================================================
   进度
   ========================================================= */

.progress-bg {

    width: 100%;

    height: 11px;

    border-radius: 20px;

    overflow: hidden;

    background:
        #0b1018;

}

.progress-bar {

    width: 0%;

    height: 100%;

    background:
        #1aa6b2;

    transition:
        width .3s ease;

}

.progress-info {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-top: 8px;

    font-size: 12px;

    color:
        #8793a7;

}


/* =========================================================
   右侧顶部状态
   ========================================================= */

.monitor-head {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 14px;

}

.monitor-title {

    font-size: 19px;

    font-weight: 700;

}

.monitor-status {

    display: flex;

    align-items: center;

    gap: 9px;

    padding:
        8px 13px;

    border-radius: 20px;

    background:
        #121822;

    border:
        1px solid #293343;

    color:
        #b7c1d0;

    font-size: 13px;

}


/* =========================================================
   指标卡
   ========================================================= */

.metric-grid {

    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 10px;

    margin-bottom: 10px;

}

.metric {

    min-width: 0;

    padding:
        14px;

    border-radius: 12px;

    background:
        #111722;

    border:
        1px solid #252e3d;

}

.metric-label {

    color:
        #748096;

    font-size: 11px;

    margin-bottom: 6px;

}

.metric-value {

    overflow: hidden;

    white-space: nowrap;

    text-overflow: ellipsis;

    color:
        #eef3fa;

    font-size: 16px;

    font-weight: 700;

}

.metric-sub {

    margin-top: 4px;

    color:
        #667387;

    font-size: 11px;

}


/* =========================================================
   当前话术
   ========================================================= */

.current-card {

    padding:
        17px;

    border-radius: 13px;

    background:
        #111722;

    border:
        1px solid #252e3d;

    margin-top: 10px;

}

.current-head {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 10px;

    margin-bottom: 11px;

}

.current-title {

    font-size: 16px;

    font-weight: 700;

}

.current-version {

    color:
        #7f8ca1;

    font-size: 12px;

}

.current-text {

    min-height: 100px;

    padding:
        14px;

    border-radius: 10px;

    background:
        #0b1018;

    color:
        #dce4ef;

    font-size: 16px;

    line-height: 1.85;

    white-space: pre-wrap;

}


/* =========================================================
   当前音频
   ========================================================= */

.audio-row {

    display: grid;

    grid-template-columns:
        minmax(0, 1fr)
        110px;

    gap: 10px;

    margin-top: 10px;

}

.audio-info {

    padding:
        12px;

    background:
        #111722;

    border:
        1px solid #252e3d;

    border-radius: 10px;

}

.audio-label {

    color:
        #718096;

    font-size: 11px;

}

.audio-name {

    margin-top: 4px;

    overflow: hidden;

    white-space: nowrap;

    text-overflow: ellipsis;

    font-size: 13px;

    color:
        #dfe7f1;

}

.audio-duration {

    display: flex;

    flex-direction: column;

    justify-content: center;

    align-items: center;

    border-radius: 10px;

    background:
        #111722;

    border:
        1px solid #252e3d;

}

.audio-duration strong {

    font-size: 18px;

}

.audio-duration span {

    margin-top: 3px;

    color:
        #718096;

    font-size: 10px;

}


/* =========================================================
   直播时长 / 轮次
   ========================================================= */

.big-stats {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 10px;

    margin-top: 10px;

}

.big-stat {

    padding:
        14px;

    border-radius: 12px;

    background:
        #111722;

    border:
        1px solid #252e3d;

}

.big-stat-label {

    color:
        #718096;

    font-size: 11px;

}

.big-stat-value {

    margin-top: 4px;

    font-size: 23px;

    font-weight: 700;

}


/* =========================================================
   直播进度
   ========================================================= */

.monitor-progress {

    margin-top: 10px;

    padding:
        14px;

    border-radius: 12px;

    background:
        #111722;

    border:
        1px solid #252e3d;

}

.monitor-progress-head {

    display: flex;

    justify-content: space-between;

    margin-bottom: 8px;

    font-size: 12px;

    color:
        #8290a5;

}

.live-progress-bg {

    width: 100%;

    height: 14px;

    background:
        #080d14;

    border-radius: 20px;

    overflow: hidden;

}

.live-progress-bar {

    width: 0%;

    height: 100%;

    background:
        #1677ff;

    transition:
        width .25s ease;

}


/* =========================================================
   日志
   ========================================================= */

.log-card {

    margin-top: 10px;

    border-radius: 12px;

    overflow: hidden;

    border:
        1px solid #252e3d;

}

.log-head {

    padding:
        11px 13px;

    background:
        #151c27;

    color:
        #aeb9c8;

    font-size: 13px;

    font-weight: 700;

}

.log-box {

    height: 205px;

    overflow-y: auto;

    padding:
        12px;

    background:
        #06090e;

    color:
        #9ba7b8;

    font-family:
        Consolas,
        "Courier New",
        monospace;

    font-size: 12px;

    line-height: 1.65;

    white-space: pre-wrap;

}


/* =========================================================
   异常
   ========================================================= */

.alert-card {

    margin-top: 10px;

    padding:
        13px;

    border-radius: 12px;

    background:
        #10161f;

    border:
        1px solid #283241;

}

.alert-card.ok {

    border-color:
        #224a36;

    background:
        #0d1813;

}

.alert-card.warning {

    border-color:
        #5b4821;

    background:
        #18150d;

}

.alert-card.error {

    border-color:
        #5b2730;

    background:
        #190e11;

}

.alert-title {

    font-size: 12px;

    font-weight: 700;

    margin-bottom: 4px;

}

.alert-message {

    color:
        #8996aa;

    font-size: 12px;

}


/* =========================================================
   商品话术
   ========================================================= */

.script-box {

    max-height: 380px;

    overflow-y: auto;

    padding:
        14px;

    border-radius: 10px;

    background:
        #0d121b;

    color:
        #cfd8e5;

    font-size: 13px;

    line-height: 1.8;

    white-space: pre-wrap;

}


/* =========================================================
   添加商品
   ========================================================= */

.editor {

    display: none;

    margin-top: 10px;

    padding:
        12px;

    border-radius: 10px;

    background:
        #0d121b;

    border:
        1px solid #283241;

}

.editor-title {

    margin-bottom: 9px;

    font-size: 13px;

    font-weight: 700;

}


/* =========================================================
   响应式
   ========================================================= */

@media (max-width: 1050px) {

    body {
        overflow: auto;
    }

    .app {
        height: auto;
        min-height: 100vh;
    }

    .main {

        grid-template-columns:
            1fr;

    }

    .left {

        border-right:
            0;

        border-bottom:
            1px solid #202735;

        overflow:
            visible;

    }

    .monitor {

        overflow:
            visible;

    }

}


@media (max-width: 650px) {

    .topbar {
        padding: 0 13px;
    }

    .brand-subtitle {
        display: none;
    }

    .left,
    .monitor {
        padding: 12px;
    }

    .metric-grid {
        grid-template-columns:
            1fr 1fr;
    }

    .live-buttons {
        grid-template-columns:
            1fr;
    }

    .info-grid {
        grid-template-columns:
            1fr;
    }

}


/* =========================================================
   状态颜色
   ========================================================= */

.status-playing {
    color:
        #35d07f !important;
}

.status-paused {
    color:
        #f5b83d !important;
}

.status-starting {
    color:
        #f5b83d !important;
}

.status-stopped {
    color:
        #7d899b !important;
}

.status-error {
    color:
        #ff5c68 !important;
}

</style>

</head>


<body>

<div class="app">


<!-- =======================================================
     顶部
     ======================================================= -->

<header class="topbar">

    <div class="brand">

        <div class="brand-icon">
            🎙️
        </div>

        <div>

            <div class="brand-title">
                AI直播控制台 V8
            </div>

            <div class="brand-subtitle">
                商品管理 · AI话术 · WAV语音 · 专业直播监控
            </div>

        </div>

    </div>

    <div
        class="clock"
        id="clock"
    >
        --
    </div>

</header>


<!-- =======================================================
     主体
     ======================================================= -->

<div class="main">


<!-- =======================================================
     左侧
     ======================================================= -->

<div class="left">


<!-- =======================================================
     直播控制 —— 最顶部
     ======================================================= -->

<div class="panel live-control">

    <div class="live-control-head">

        <div class="live-control-title">
            🎙️ 直播控制
        </div>

        <div class="live-control-status">

            <span
                class="status-dot"
                id="controlStatusDot"
            ></span>

            <span id="controlStatusText">
                已停止
            </span>

        </div>

    </div>


    <div class="live-buttons">

        <button
            class="live-btn btn-start"
            id="startLiveBtn"
            onclick="startLive()"
        >
            ▶ 开始直播
        </button>

        <button
            class="live-btn btn-pause"
            id="pauseBtn"
            onclick="pauseLive()"
        >
            ⏸ 一句话说完后暂停
        </button>

        <button
            class="live-btn btn-resume"
            id="resumeBtn"
            onclick="resumeLive()"
        >
            ▶ 继续播放
        </button>

        <button
            class="live-btn btn-stop"
            id="stopLiveBtn"
            onclick="stopLive()"
        >
            ⏹ 停止直播
        </button>

    </div>


    <div class="live-hint">

        开始直播后，当前商品将持续循环播放。
        每轮随机选择已经生成完成的完整话术版本。
        <b>不会自动切换商品，也不会自动生成新的话术或 WAV。</b>

    </div>

</div>


<!-- =======================================================
     商品
     ======================================================= -->

<div class="panel">

    <div class="panel-title">

        <h2>
            📦 商品管理
        </h2>

        <span class="small">
            当前直播商品
        </span>

    </div>


    <select id="product"></select>


    <div class="row" style="margin-top:9px;">

        <button
            class="tool-btn btn-purple"
            onclick="addProduct()"
            style="flex:1;"
        >
            ➕ 添加商品
        </button>

        <button
            class="tool-btn"
            onclick="deleteProduct()"
            style="flex:1;background:#54222a;color:#ffb7bd;border-color:#71313b;"
        >
            🗑️ 删除商品
        </button>

    </div>


    <div class="tool-grid">

        <button
            class="tool-btn"
            onclick="openLocalFolder('script')"
        >
            📂 打开本地话术
        </button>

        <button
            class="tool-btn"
            onclick="openLocalFolder('wav')"
        >
            🔊 打开本地 WAV
        </button>

    </div>


    <div
        class="editor"
        id="productEditor"
    >

        <div class="editor-title">
            ➕ 添加新商品
        </div>

        <div class="info-grid">

            <input
                id="newProductName"
                placeholder="商品名称 *"
            >

            <input
                id="newProductPrice"
                placeholder="价格，例如：99元"
            >

            <input
                id="newProductSpecification"
                placeholder="规格/型号/数量"
            >

            <input
                id="newProductTarget"
                placeholder="适用人群"
            >

            <input
                id="newProductScene"
                placeholder="使用场景"
            >

        </div>


        <textarea
            id="newProductSellingPoints"
            placeholder="核心卖点（每条一行，填写真实资料）"
            style="height:85px;"
        ></textarea>


        <textarea
            id="newProductNotes"
            placeholder="注意事项"
            style="height:65px;"
        ></textarea>


        <textarea
            id="newProductExtra"
            placeholder="其他补充信息"
            style="height:80px;"
        ></textarea>


        <textarea
            id="newProductScript"
            placeholder="初始话术，可留空"
            style="height:120px;"
        ></textarea>


        <div class="row" style="margin-top:9px;">

            <button
                class="tool-btn btn-purple"
                onclick="confirmAddProduct()"
            >
                确认添加
            </button>

            <button
                class="tool-btn"
                onclick="cancelAddProduct()"
            >
                取消
            </button>

        </div>

    </div>

</div>


<!-- =======================================================
     商品资料
     ======================================================= -->

<div
    class="panel"
    id="productInfoPanel"
    style="display:none;"
>

    <div class="panel-title">

        <h2>
            📋 商品资料
        </h2>

        <span class="small">
            AI创作事实依据
        </span>

    </div>


    <div class="info-grid">

        <input
            id="infoPrice"
            placeholder="价格"
        >

        <input
            id="infoSpecification"
            placeholder="规格/型号/数量"
        >

        <input
            id="infoTarget"
            placeholder="适用人群"
        >

        <input
            id="infoScene"
            placeholder="使用场景"
        >

    </div>


    <textarea
        id="infoSellingPoints"
        placeholder="核心卖点"
        style="height:80px;"
    ></textarea>


    <textarea
        id="infoNotes"
        placeholder="注意事项"
        style="height:65px;"
    ></textarea>


    <textarea
        id="infoExtra"
        placeholder="其他补充信息"
        style="height:75px;"
    ></textarea>


    <button
        class="tool-btn btn-purple"
        onclick="saveProductInfo()"
        style="margin-top:8px;width:100%;"
    >
        💾 保存商品资料
    </button>


    <div class="ai-message">
        修改商品资料后会自动清理旧 WAV，请重新生成 WAV。
    </div>

</div>


<!-- =======================================================
     AI话术
     ======================================================= -->

<div class="panel">

    <div class="panel-title">

        <h2>
            🧠 AI话术
        </h2>

        <span class="small">
            DeepSeek
        </span>

    </div>


    <div class="ai-grid">

        <select
            id="scriptStyle"
        >

            <option value="高转化直播型">
                🔥 高转化直播型
            </option>

            <option value="自然聊天型">
                💬 自然聊天型
            </option>

            <option value="强种草型">
                🌱 强种草型
            </option>

        </select>


        <select
            id="scriptVersion"
        >

            <option value="高转化直播型">
                当前：高转化直播型
            </option>

        </select>


        <button
            class="btn-purple"
            id="refreshScriptBtn"
            onclick="regenerateScript()"
        >
            🧠 生成当前风格
        </button>


        <button
            class="btn-purple"
            id="generateAllScriptBtn"
            onclick="regenerateAllScripts()"
        >
            🧠 一键生成3套
        </button>

    </div>


    <div
        class="ai-message"
        id="aiMessage"
    >
        商品资料会作为 DeepSeek 创作的事实依据。
    </div>

</div>


<!-- =======================================================
     WAV
     ======================================================= -->

<div class="panel">

    <div class="panel-title">

        <h2>
            🔊 WAV语音
        </h2>

        <span class="small">
            CosyVoice3
        </span>

    </div>


    <button
        class="tool-btn btn-cyan"
        id="ttsBtn"
        onclick="generateAllWav()"
        style="width:100%;margin-bottom:10px;"
    >
        🎙️ 生成全部 WAV
    </button>


    <div class="progress-bg">

        <div
            class="progress-bar"
            id="ttsProgressBar"
        ></div>

    </div>


    <div class="progress-info">

        <span id="ttsProgressText">
            0 / 0
        </span>

        <span id="ttsMessage">
            等待生成
        </span>

    </div>

</div>


<!-- =======================================================
     当前商品话术
     ======================================================= -->

<div class="panel">

    <div class="panel-title">

        <h2>
            📝 商品话术
        </h2>

        <span class="small">
            script.txt
        </span>

    </div>


    <div
        class="script-box"
        id="script"
    >
        请选择商品
    </div>

</div>


</div>


<!-- =======================================================
     右侧监控
     ======================================================= -->

<div class="monitor">


<div class="monitor-head">

    <div class="monitor-title">
        📡 直播实时监控
    </div>

    <div class="monitor-status">

        <span
            class="status-dot"
            id="monitorStatusDot"
        ></span>

        <span id="monitorStatusText">
            ⏹ 已停止
        </span>

    </div>

</div>


<!-- =======================================================
     核心指标
     ======================================================= -->

<div class="metric-grid">

    <div class="metric">

        <div class="metric-label">
            当前商品
        </div>

        <div
            class="metric-value"
            id="monitorProduct"
        >
            -
        </div>

        <div class="metric-sub">
            当前直播商品
        </div>

    </div>


    <div class="metric">

        <div class="metric-label">
            当前话术
        </div>

        <div
            class="metric-value"
            id="monitorSegment"
        >
            -
        </div>

        <div class="metric-sub">
            当前播放段
        </div>

    </div>


    <div class="metric">

        <div class="metric-label">
            话术版本
        </div>

        <div
            class="metric-value"
            id="monitorVersion"
        >
            -
        </div>

        <div class="metric-sub">
            随机版本
        </div>

    </div>


    <div class="metric">

        <div class="metric-label">
            当前音频
        </div>

        <div
            class="metric-value"
            id="monitorAudio"
        >
            -
        </div>

        <div class="metric-sub">
            WAV文件
        </div>

    </div>


    <div class="metric">

        <div class="metric-label">
            当前进度
        </div>

        <div
            class="metric-value"
            id="monitorProgress"
        >
            0 / 0
        </div>

        <div class="metric-sub">
            当前话术版本
        </div>

    </div>


    <div class="metric">

        <div class="metric-label">
            当前音频时长
        </div>

        <div
            class="metric-value"
            id="monitorAudioDuration"
        >
            0.00 秒
        </div>

        <div class="metric-sub">
            当前 WAV
        </div>

    </div>

</div>


<!-- =======================================================
     直播时长 / 循环轮数
     ======================================================= -->

<div class="big-stats">

    <div class="big-stat">

        <div class="big-stat-label">
            ⏱ 总直播时长
        </div>

        <div
            class="big-stat-value"
            id="liveDuration"
        >
            00:00:00
        </div>

    </div>


    <div class="big-stat">

        <div class="big-stat-label">
            🔄 已循环轮数
        </div>

        <div
            class="big-stat-value"
            id="roundNumber"
        >
            0
        </div>

    </div>

</div>


<!-- =======================================================
     当前话术
     ======================================================= -->

<div class="current-card">

    <div class="current-head">

        <div class="current-title">
            🎧 当前正在播放
        </div>

        <div
            class="current-version"
            id="currentVersion"
        >
            -
        </div>

    </div>


    <div
        class="current-text"
        id="currentText"
    >
        当前没有正在播放的话术。
    </div>


    <div class="audio-row">

        <div class="audio-info">

            <div class="audio-label">
                当前音频文件
            </div>

            <div
                class="audio-name"
                id="currentAudio"
            >
                -
            </div>

        </div>


        <div class="audio-duration">

            <strong
                id="currentAudioDuration"
            >
                0.00
            </strong>

            <span>
                秒
            </span>

        </div>

    </div>

</div>


<!-- =======================================================
     播放进度
     ======================================================= -->

<div class="monitor-progress">

    <div class="monitor-progress-head">

        <span>
            📊 当前话术播放进度
        </span>

        <span
            id="liveProgressText"
        >
            0%
        </span>

    </div>


    <div class="live-progress-bg">

        <div
            class="live-progress-bar"
            id="liveProgressBar"
        ></div>

    </div>

</div>


<!-- =======================================================
     状态消息
     ======================================================= -->

<div class="alert-card ok" id="alertCard">

    <div
        class="alert-title"
        id="alertTitle"
    >
        ✓ 系统正常
    </div>

    <div
        class="alert-message"
        id="alertMessage"
    >
        当前没有异常。
    </div>

</div>


<!-- =======================================================
     实时日志
     ======================================================= -->

<div class="log-card">

    <div class="log-head">
        📋 直播实时运行日志
    </div>

    <div
        class="log-box"
        id="liveLogs"
    >
        暂无日志
    </div>

</div>


</div>

</div>

</div>


<script>

/* =========================================================
   全局状态
   ========================================================= */

let currentState = {};

let currentTtsState = {};

let currentAiState = {
    status: "idle"
};


/* =========================================================
   API
   ========================================================= */

async function api(
    url,
    options = {}
) {

    const res = await fetch(
        url,
        {
            headers: {
                "Content-Type":
                    "application/json"
            },

            ...options
        }
    );

    const data =
        await res.json();

    if (!res.ok) {

        throw new Error(
            data.detail ||
            "操作失败"
        );

    }

    return data;

}


/* =========================================================
   状态文字
   ========================================================= */

function statusText(
    status
) {

    const map = {

        starting:
            "🟡 正在启动",

        stopped:
            "⏹ 已停止",

        playing:
            "🟢 播放中",

        pause_requested:
            "🟡 当前一句结束后暂停",

        paused:
            "⏸ 已暂停"

    };

    return (
        map[status]
        ||
        status
        ||
        "-"
    );

}


/* =========================================================
   状态 CSS
   ========================================================= */

function statusClass(
    status
) {

    if (
        status === "playing"
    ) {

        return "playing";

    }

    if (
        status === "paused"
        ||
        status === "pause_requested"
    ) {

        return "paused";

    }

    if (
        status === "starting"
    ) {

        return "starting";

    }

    return "";

}


/* =========================================================
   时间格式
   ========================================================= */

function formatDuration(
    seconds
) {

    seconds =
        Math.max(
            0,
            Math.floor(
                Number(seconds)
                || 0
            )
        );

    const h =
        Math.floor(
            seconds / 3600
        );

    const m =
        Math.floor(
            (seconds % 3600) / 60
        );

    const s =
        seconds % 60;

    return (
        String(h).padStart(2, "0")
        + ":"
        +
        String(m).padStart(2, "0")
        + ":"
        +
        String(s).padStart(2, "0")
    );

}


/* =========================================================
   当前直播时长
   ========================================================= */

function updateLiveDuration() {

    const started =
        Number(
            currentState.live_started_at
        );

    if (!started) {

        document.getElementById(
            "liveDuration"
        ).textContent =
            "00:00:00";

        return;

    }

    let end;

    if (
        currentState.status ===
        "stopped"
        &&
        currentState.live_stopped_at
    ) {

        end =
            Number(
                currentState.live_stopped_at
            );

    } else {

        end =
            Date.now() / 1000;

    }

    const seconds =
        Math.max(
            0,
            end - started
        );

    document.getElementById(
        "liveDuration"
    ).textContent =
        formatDuration(
            seconds
        );

}


/* =========================================================
   时钟
   ========================================================= */

function updateClock() {

    const now =
        new Date();

    const text =
        now.getFullYear()
        + "-"
        +
        String(
            now.getMonth() + 1
        ).padStart(2, "0")
        + "-"
        +
        String(
            now.getDate()
        ).padStart(2, "0")
        + " "
        +
        String(
            now.getHours()
        ).padStart(2, "0")
        + ":"
        +
        String(
            now.getMinutes()
        ).padStart(2, "0")
        + ":"
        +
        String(
            now.getSeconds()
        ).padStart(2, "0");

    document.getElementById(
        "clock"
    ).textContent =
        text;

}


/* =========================================================
   播放按钮
   ========================================================= */

function updateButtons(
    state
) {

    const playing =
        state.status ===
        "playing";

    const requested =
        state.status ===
        "pause_requested";

    const paused =
        state.status ===
        "paused";

    const starting =
        state.status ===
        "starting";

    const stopped =
        state.status ===
        "stopped";


    document.getElementById(
        "startLiveBtn"
    ).disabled =
        !stopped;


    document.getElementById(
        "pauseBtn"
    ).disabled =
        !playing;


    document.getElementById(
        "resumeBtn"
    ).disabled =
        !(
            paused
            ||
            requested
        );


    document.getElementById(
        "stopLiveBtn"
    ).disabled =
        stopped
        ||
        starting;

}


/* =========================================================
   状态灯
   ========================================================= */

function updateStatusIndicators(
    state
) {

    const dot1 =
        document.getElementById(
            "controlStatusDot"
        );

    const dot2 =
        document.getElementById(
            "monitorStatusDot"
        );

    const cls =
        statusClass(
            state.status
        );


    dot1.className =
        "status-dot "
        +
        cls;

    dot2.className =
        "status-dot "
        +
        cls;


    document.getElementById(
        "controlStatusText"
    ).textContent =
        statusText(
            state.status
        );


    document.getElementById(
        "monitorStatusText"
    ).textContent =
        statusText(
            state.status
        );


    const statusElement =
        document.getElementById(
            "monitorStatusText"
        );

    statusElement.className =
        "";


    if (
        state.status ===
        "playing"
    ) {

        statusElement.classList.add(
            "status-playing"
        );

    } else if (
        state.status ===
        "paused"
        ||
        state.status ===
        "pause_requested"
        ||
        state.status ===
        "starting"
    ) {

        statusElement.classList.add(
            "status-paused"
        );

    } else {

        statusElement.classList.add(
            "status-stopped"
        );

    }

}


/* =========================================================
   日志渲染
   ========================================================= */

function renderLogs(
    logs
) {

    const box =
        document.getElementById(
            "liveLogs"
        );

    const wasNearBottom =
        box.scrollHeight
        -
        box.scrollTop
        -
        box.clientHeight
        <
        60;


    if (
        !logs
        ||
        !logs.length
    ) {

        box.textContent =
            "暂无日志";

        return;

    }


    box.textContent =
        logs.join("\n");


    if (wasNearBottom) {

        box.scrollTop =
            box.scrollHeight;

    }

}


/* =========================================================
   状态渲染
   ========================================================= */

function render(
    state
) {

    currentState =
        state;


    updateButtons(
        state
    );


    updateStatusIndicators(
        state
    );


    document.getElementById(
        "monitorProduct"
    ).textContent =
        state.product
        ||
        "-";


    document.getElementById(
        "monitorSegment"
    ).textContent =
        state.current_title
        ||
        "-";


    document.getElementById(
        "monitorVersion"
    ).textContent =
        state.current_version
        ||
        "-";


    document.getElementById(
        "monitorAudio"
    ).textContent =
        state.current_audio
        ||
        "-";


    document.getElementById(
        "monitorProgress"
    ).textContent =
        `${state.current_index || 0} / ${state.total || 0}`;


    const audioDuration =
        Number(
            state.current_audio_duration
        )
        ||
        0;


    document.getElementById(
        "monitorAudioDuration"
    ).textContent =
        audioDuration.toFixed(
            2
        )
        +
        " 秒";


    document.getElementById(
        "currentAudio"
    ).textContent =
        state.current_audio
        ||
        "-";


    document.getElementById(
        "currentAudioDuration"
    ).textContent =
        audioDuration.toFixed(
            2
        );


    document.getElementById(
        "roundNumber"
    ).textContent =
        state.round_no
        ||
        0;


    document.getElementById(
        "currentVersion"
    ).textContent =
        state.current_version
        ||
        "-";


    document.getElementById(
        "currentText"
    ).textContent =
        state.current_text
        ||
        "当前没有正在播放的话术。";


    const percent =
        Number(
            state.progress_percent
        )
        ||
        0;


    document.getElementById(
        "liveProgressBar"
    ).style.width =
        Math.max(
            0,
            Math.min(
                100,
                percent
            )
        )
        +
        "%";


    document.getElementById(
        "liveProgressText"
    ).textContent =
        percent
        +
        "%";


    renderLogs(
        state.logs
    );


    renderAlert(
        state
    );


    updateLiveDuration();

}


/* =========================================================
   异常提示
   ========================================================= */

function renderAlert(
    state
) {

    const card =
        document.getElementById(
            "alertCard"
        );

    const title =
        document.getElementById(
            "alertTitle"
        );

    const message =
        document.getElementById(
            "alertMessage"
        );


    let level =
        state.alert_level
        ||
        "ok";

    let titleText =
        "✓ 系统正常";

    let messageText =
        "当前没有异常。";


    if (
        level === "error"
        ||
        currentTtsState.status ===
        "error"
        ||
        currentAiState.status ===
        "error"
    ) {

        level =
            "error";

        titleText =
            "⚠ 发现异常";


        if (
            state.alert_level ===
            "error"
            &&
            state.alert_message
        ) {

            messageText =
                state.alert_message;

        } else if (
            currentTtsState.status ===
            "error"
        ) {

            messageText =
                "WAV生成失败："
                +
                (
                    currentTtsState.error
                    ||
                    currentTtsState.message
                    ||
                    "未知错误"
                );

        } else if (
            currentAiState.status ===
            "error"
        ) {

            messageText =
                "AI话术生成失败："
                +
                (
                    currentAiState.error
                    ||
                    currentAiState.message
                    ||
                    "未知错误"
                );

        }

    } else if (
        level === "warning"
        ||
        state.status ===
        "pause_requested"
    ) {

        level =
            "warning";

        titleText =
            "⚠ 当前状态提示";

        messageText =
            state.alert_message
            ||
            state.message
            ||
            "当前正在等待操作。";

    }


    card.className =
        "alert-card "
        +
        level;

    title.textContent =
        titleText;

    message.textContent =
        messageText;

}


/* =========================================================
   商品添加
   ========================================================= */

function addProduct() {

    document.getElementById(
        "productEditor"
    ).style.display =
        "block";

    document.getElementById(
        "newProductName"
    ).focus();

}


function cancelAddProduct() {

    document.getElementById(
        "productEditor"
    ).style.display =
        "none";


    [

        "newProductName",

        "newProductPrice",

        "newProductSpecification",

        "newProductTarget",

        "newProductScene",

        "newProductSellingPoints",

        "newProductNotes",

        "newProductExtra",

        "newProductScript"

    ].forEach(
        id => {

            document.getElementById(
                id
            ).value =
                "";

        }
    );

}


function addInfoField(
    id
) {

    return document.getElementById(
        id
    ).value.trim();

}


async function confirmAddProduct() {

    const product =
        addInfoField(
            "newProductName"
        );

    const script =
        addInfoField(
            "newProductScript"
        );


    if (!product) {

        alert(
            "请输入商品名称"
        );

        return;

    }


    try {

        await api(
            "/api/product/create",
            {

                method:
                    "POST",

                body:
                    JSON.stringify(

                        {

                            product,

                            script,

                            price:
                                addInfoField(
                                    "newProductPrice"
                                ),

                            specification:
                                addInfoField(
                                    "newProductSpecification"
                                ),

                            selling_points:
                                addInfoField(
                                    "newProductSellingPoints"
                                ),

                            target_people:
                                addInfoField(
                                    "newProductTarget"
                                ),

                            usage_scene:
                                addInfoField(
                                    "newProductScene"
                                ),

                            notes:
                                addInfoField(
                                    "newProductNotes"
                                ),

                            extra:
                                addInfoField(
                                    "newProductExtra"
                                )

                        }

                    )

            }
        );


        cancelAddProduct();

        await loadProducts(
            product
        );

        alert(
            `商品「${product}」添加成功`
        );

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   商品资料
   ========================================================= */

async function loadProductInfo() {

    const product =
        document.getElementById(
            "product"
        ).value;

    const panel =
        document.getElementById(
            "productInfoPanel"
        );


    if (!product) {

        panel.style.display =
            "none";

        return;

    }


    try {

        const info =
            await api(
                "/api/product/"
                +
                encodeURIComponent(
                    product
                )
                +
                "/info"
            );


        panel.style.display =
            "block";


        document.getElementById(
            "infoPrice"
        ).value =
            info.price
            ||
            "";


        document.getElementById(
            "infoSpecification"
        ).value =
            info.specification
            ||
            "";


        document.getElementById(
            "infoTarget"
        ).value =
            info.target_people
            ||
            "";


        document.getElementById(
            "infoScene"
        ).value =
            info.usage_scene
            ||
            "";


        document.getElementById(
            "infoSellingPoints"
        ).value =
            info.selling_points
            ||
            "";


        document.getElementById(
            "infoNotes"
        ).value =
            info.notes
            ||
            "";


        document.getElementById(
            "infoExtra"
        ).value =
            info.extra
            ||
            "";

    } catch (e) {

        panel.style.display =
            "none";

        console.error(
            e
        );

    }

}


async function saveProductInfo() {

    const product =
        document.getElementById(
            "product"
        ).value;


    if (!product) {

        return;

    }


    try {

        const data =
            await api(
                "/api/product/info",
                {

                    method:
                        "POST",

                    body:
                        JSON.stringify(

                            {

                                product,

                                price:
                                    addInfoField(
                                        "infoPrice"
                                    ),

                                specification:
                                    addInfoField(
                                        "infoSpecification"
                                    ),

                                target_people:
                                    addInfoField(
                                        "infoTarget"
                                    ),

                                usage_scene:
                                    addInfoField(
                                        "infoScene"
                                    ),

                                selling_points:
                                    addInfoField(
                                        "infoSellingPoints"
                                    ),

                                notes:
                                    addInfoField(
                                        "infoNotes"
                                    ),

                                extra:
                                    addInfoField(
                                        "infoExtra"
                                    )

                            }

                        )

                }
            );


        await loadScript();

        await refreshTtsState();

        document.getElementById(
            "aiMessage"
        ).textContent =
            "📦 "
            +
            data.message;


        alert(
            data.message
        );

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   删除商品
   ========================================================= */

async function deleteProduct() {

    const product =
        document.getElementById(
            "product"
        ).value;


    if (!product) {

        alert(
            "请先选择商品"
        );

        return;

    }


    if (
        !confirm(

            `确定删除商品「${product}」吗？\n\n`
            +
            `将同时删除：\n`
            +
            `• 商品话术文件夹\n`
            +
            `• 商品 WAV 文件夹\n`
            +
            `• 3套 AI 话术版本\n\n`
            +
            `删除后无法恢复。`

        )
    ) {

        return;

    }


    try {

        await api(

            "/api/product/"
            +
            encodeURIComponent(
                product
            ),

            {
                method:
                    "DELETE"
            }

        );


        await loadProducts();

        alert(
            `商品「${product}」已删除`
        );

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   打开本地目录
   ========================================================= */

async function openLocalFolder(
    kind
) {

    const product =
        document.getElementById(
            "product"
        ).value;


    if (!product) {

        alert(
            "请先选择商品"
        );

        return;

    }


    try {

        const data =
            await api(
                "/api/local/open",
                {

                    method:
                        "POST",

                    body:
                        JSON.stringify(
                            {
                                product,
                                kind
                            }
                        )

                }
            );


        document.getElementById(
            "alertMessage"
        ).textContent =
            "📂 "
            +
            data.message;

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   商品列表
   ========================================================= */

async function loadProducts(
    preferredProduct = ""
) {

    try {

        const data =
            await api(
                "/api/products"
            );


        const select =
            document.getElementById(
                "product"
            );


        select.innerHTML =
            "";


        data.products.forEach(
            product => {

                const option =
                    document.createElement(
                        "option"
                    );

                option.value =
                    product;

                option.textContent =
                    product;

                select.appendChild(
                    option
                );

            }
        );


        if (
            preferredProduct
            &&
            data.products.includes(
                preferredProduct
            )
        ) {

            select.value =
                preferredProduct;

        }


        if (
            data.products.length
        ) {

            await loadScript();

            await loadScriptVersions();

            await loadProductInfo();

            await refreshTtsState();

        }

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   读取话术
   ========================================================= */

async function loadScript() {

    const product =
        document.getElementById(
            "product"
        ).value;


    if (!product) {

        return;

    }


    try {

        const data =
            await api(

                "/api/product/"
                +
                encodeURIComponent(
                    product
                )

            );


        document.getElementById(
            "script"
        ).textContent =
            data.script;

    } catch (e) {

        document.getElementById(
            "script"
        ).textContent =
            "读取话术失败："
            +
            e.message;

    }

}


/* =========================================================
   AI生成当前风格
   ========================================================= */

async function regenerateScript() {

    const product =
        document.getElementById(
            "product"
        ).value;

    const style =
        document.getElementById(
            "scriptStyle"
        ).value;


    if (!product) {

        alert(
            "请先选择商品"
        );

        return;

    }


    if (
        currentState.status ===
        "playing"
        ||
        currentState.status ===
        "pause_requested"
        ||
        currentState.status ===
        "paused"
    ) {

        alert(
            "直播正在运行，请先停止直播后再生成 AI 话术。"
        );

        return;

    }


    if (
        currentTtsState.status ===
        "loading"
        ||
        currentTtsState.status ===
        "generating"
    ) {

        alert(
            "WAV 正在生成，请等待完成后再生成 AI 话术。"
        );

        return;

    }


    const btn =
        document.getElementById(
            "refreshScriptBtn"
        );

    const btn2 =
        document.getElementById(
            "generateAllScriptBtn"
        );


    btn.disabled =
        true;

    btn2.disabled =
        true;

    btn.textContent =
        "🧠 正在生成...";


    document.getElementById(
        "aiMessage"
    ).textContent =
        "🧠 正在调用 DeepSeek，请稍候...";


    try {

        await api(

            "/api/script/regenerate",

            {

                method:
                    "POST",

                body:
                    JSON.stringify(
                        {
                            product,
                            style
                        }
                    )

            }

        );


        await refreshAiState();

    } catch (e) {

        alert(
            e.message
        );

        btn.disabled =
            false;

        btn2.disabled =
            false;

        btn.textContent =
            "🧠 生成当前风格";

    }

}


/* =========================================================
   AI生成全部
   ========================================================= */

async function regenerateAllScripts() {

    const product =
        document.getElementById(
            "product"
        ).value;


    if (!product) {

        alert(
            "请先选择商品"
        );

        return;

    }


    if (
        currentState.status ===
        "playing"
        ||
        currentState.status ===
        "pause_requested"
        ||
        currentState.status ===
        "paused"
    ) {

        alert(
            "直播正在运行，请先停止直播后再生成 AI 话术。"
        );

        return;

    }


    if (
        currentTtsState.status ===
        "loading"
        ||
        currentTtsState.status ===
        "generating"
    ) {

        alert(
            "WAV 正在生成，请等待完成后再生成 AI 话术。"
        );

        return;

    }


    const b1 =
        document.getElementById(
            "refreshScriptBtn"
        );

    const b2 =
        document.getElementById(
            "generateAllScriptBtn"
        );


    b1.disabled =
        true;

    b2.disabled =
        true;

    b2.textContent =
        "🧠 正在生成3套...";


    try {

        await api(

            "/api/script/regenerate-all",

            {

                method:
                    "POST",

                body:
                    JSON.stringify(
                        {
                            product,
                            style:
                                "高转化直播型"
                        }
                    )

            }

        );


        await refreshAiState();

    } catch (e) {

        alert(
            e.message
        );

        b1.disabled =
            false;

        b2.disabled =
            false;

        b2.textContent =
            "🧠 一键生成3套";

    }

}


/* =========================================================
   AI状态
   ========================================================= */

async function refreshAiState() {

    try {

        const state =
            await api(
                "/api/script/status"
            );


        currentAiState =
            state;


        const b1 =
            document.getElementById(
                "refreshScriptBtn"
            );

        const b2 =
            document.getElementById(
                "generateAllScriptBtn"
            );

        const ttsBtn =
            document.getElementById(
                "ttsBtn"
            );

        const msg =
            document.getElementById(
                "aiMessage"
            );


        if (
            state.status ===
            "generating"
        ) {

            b1.disabled =
                true;

            b2.disabled =
                true;

            ttsBtn.disabled =
                true;

            b1.textContent =
                "🧠 正在生成...";

            b2.textContent =
                state.total > 1
                ?
                `🧠 ${state.current} / ${state.total}`
                :
                "🧠 生成中...";

            msg.textContent =
                "🧠 "
                +
                (
                    state.message
                    ||
                    "DeepSeek 正在生成话术..."
                );

        } else if (
            state.status ===
            "completed"
        ) {

            b1.disabled =
                false;

            b2.disabled =
                false;

            ttsBtn.disabled =
                false;

            b1.textContent =
                "🧠 生成当前风格";

            b2.textContent =
                "🧠 一键生成3套";

            msg.textContent =
                "✅ "
                +
                state.message;

            await loadScript();

            await loadScriptVersions();

            await refreshTtsState();

        } else if (
            state.status ===
            "error"
        ) {

            b1.disabled =
                false;

            b2.disabled =
                false;

            ttsBtn.disabled =
                false;

            b1.textContent =
                "🧠 生成当前风格";

            b2.textContent =
                "🧠 一键生成3套";

            msg.textContent =
                "❌ "
                +
                (
                    state.error
                    ||
                    "AI话术生成失败"
                );

        }

        renderAlert(
            currentState
        );

    } catch (e) {

        console.error(
            e
        );

    }

}


/* =========================================================
   话术版本
   ========================================================= */

async function loadScriptVersions() {

    const product =
        document.getElementById(
            "product"
        ).value;


    if (!product) {

        return;

    }


    try {

        const data =
            await api(

                "/api/script/versions/"
                +
                encodeURIComponent(
                    product
                )

            );


        const select =
            document.getElementById(
                "scriptVersion"
            );


        select.innerHTML =
            "";


        data.versions.forEach(
            v => {

                const o =
                    document.createElement(
                        "option"
                    );

                o.value =
                    v.style;

                o.textContent =
                    (
                        v.exists
                        ?
                        (
                            v.active
                            ?
                            "✓ "
                            :
                            ""
                        )
                        :
                        "— "
                    )
                    +
                    v.style
                    +
                    (
                        v.exists
                        ?
                        ""
                        :
                        "（未生成）"
                    );


                o.disabled =
                    !v.exists;


                if (
                    v.active
                ) {

                    o.selected =
                        true;

                }


                select.appendChild(
                    o
                );

            }
        );

    } catch (e) {

        console.error(
            e
        );

    }

}


/* =========================================================
   切换话术版本
   ========================================================= */

async function selectScriptVersion() {

    const product =
        document.getElementById(
            "product"
        ).value;

    const style =
        document.getElementById(
            "scriptVersion"
        ).value;


    if (
        !product
        ||
        !style
    ) {

        return;

    }


    try {

        await api(

            "/api/script/select",

            {

                method:
                    "POST",

                body:
                    JSON.stringify(
                        {
                            product,
                            style
                        }
                    )

            }

        );


        document.getElementById(
            "aiMessage"
        ).textContent =
            "✅ 已切换到「"
            +
            style
            +
            "」，旧 WAV 已清理，请重新生成 WAV。";


        await loadScript();

        await loadScriptVersions();

        await refreshTtsState();

    } catch (e) {

        alert(
            e.message
        );

        await loadScriptVersions();

    }

}


/* =========================================================
   WAV生成
   ========================================================= */

async function generateAllWav() {

    const product =
        document.getElementById(
            "product"
        ).value;


    if (!product) {

        alert(
            "请先选择商品"
        );

        return;

    }


    if (
        currentState.status ===
        "playing"
        ||
        currentState.status ===
        "pause_requested"
        ||
        currentState.status ===
        "paused"
    ) {

        alert(
            "直播正在运行，请先停止直播。"
        );

        return;

    }


    if (
        currentAiState.status ===
        "generating"
    ) {

        alert(
            "AI 话术正在生成，请等待完成后再生成 WAV。"
        );

        return;

    }


    try {

        await api(

            "/api/tts/generate",

            {

                method:
                    "POST",

                body:
                    JSON.stringify(
                        {
                            product
                        }
                    )

            }

        );


        await refreshTtsState();

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   TTS状态
   ========================================================= */

async function refreshTtsState() {

    try {

        const state =
            await api(
                "/api/tts/status"
            );


        renderTts(
            state
        );


    } catch (e) {

        console.error(
            e
        );

    }

}


function renderTts(
    state
) {

    currentTtsState =
        state;


    const total =
        state.total
        ||
        0;

    const current =
        state.current_index
        ||
        0;

    const percent =
        state.percent
        ||
        0;


    document.getElementById(
        "ttsProgressBar"
    ).style.width =
        percent
        +
        "%";


    document.getElementById(
        "ttsProgressText"
    ).textContent =
        `${current} / ${total} (${percent}%)`;


    document.getElementById(
        "ttsMessage"
    ).textContent =
        state.message
        ||
        "-";


    const btn =
        document.getElementById(
            "ttsBtn"
        );


    if (
        state.status ===
        "loading"
        ||
        state.status ===
        "generating"
    ) {

        btn.disabled =
            true;

    } else {

        btn.disabled =
            false;

    }


    if (
        currentAiState.status ===
        "generating"
    ) {

        btn.disabled =
            true;

    }


    if (
        state.status ===
        "completed"
    ) {

        document.getElementById(
            "ttsMessage"
        ).textContent =
            "✅ "
            +
            state.message;

    }


    if (
        state.status ===
        "error"
    ) {

        document.getElementById(
            "ttsMessage"
        ).textContent =
            "❌ "
            +
            (
                state.error
                ||
                "生成失败"
            );

    }


    renderAlert(
        currentState
    );

}


/* =========================================================
   开始直播
   ========================================================= */

async function startLive() {

    const product =
        document.getElementById(
            "product"
        ).value;


    if (!product) {

        alert(
            "请先选择商品"
        );

        return;

    }


    try {

        const state =
            await api(

                "/api/play/start",

                {

                    method:
                        "POST",

                    body:
                        JSON.stringify(
                            {
                                product
                            }
                        )

                }

            );


        render(
            state
        );

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   暂停
   ========================================================= */

async function pauseLive() {

    try {

        const state =
            await api(
                "/api/play/pause",
                {
                    method:
                        "POST"
                }
            );


        render(
            state
        );

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   继续
   ========================================================= */

async function resumeLive() {

    try {

        const state =
            await api(
                "/api/play/resume",
                {
                    method:
                        "POST"
                }
            );


        render(
            state
        );

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   停止
   ========================================================= */

async function stopLive() {

    try {

        const state =
            await api(
                "/api/play/stop",
                {
                    method:
                        "POST"
                }
            );


        render(
            state
        );

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   播放状态刷新
   ========================================================= */

async function refreshState() {

    try {

        const state =
            await api(
                "/api/status"
            );


        render(
            state
        );

    } catch (e) {

        console.error(
            e
        );

    }

}


/* =========================================================
   商品切换
   ========================================================= */

document.getElementById(
    "product"
).addEventListener(
    "change",
    async () => {

        await loadScript();

        await loadScriptVersions();

        await loadProductInfo();

        await refreshTtsState();

    }
);


/* =========================================================
   话术版本切换
   ========================================================= */

document.getElementById(
    "scriptVersion"
).addEventListener(
    "change",
    selectScriptVersion
);


/* =========================================================
   初始化
   ========================================================= */

async function init() {

    await loadProducts();

    await refreshState();

    await refreshTtsState();

    await refreshAiState();

    updateClock();

}


init();


/* =========================================================
   定时刷新
   ========================================================= */

setInterval(
    refreshState,
    500
);

setInterval(
    refreshTtsState,
    500
);

setInterval(
    refreshAiState,
    500
);

setInterval(
    updateLiveDuration,
    1000
);

setInterval(
    updateClock,
    1000
);

</script>

</body>

</html>
"""


# =========================================================
# 首页
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def index():

    return HTML


# =========================================================
# 商品列表
# =========================================================

@app.get("/api/products")
def products():

    return {
        "products":
            get_products()
    }


# =========================================================
# 创建商品
# =========================================================

@app.post(
    "/api/product/create"
)
def api_product_create(
    req: ProductCreateRequest
):

    try:

        info = req.model_dump()

        product = create_product(

            req.product,

            req.script,

            info

        )

        return {

            "ok":
                True,

            "product":
                product,

            "message":
                f"商品「{product}」添加成功。",

            "info":
                get_product_info(
                    product
                )

        }

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# =========================================================
# 商品资料
# =========================================================

@app.get(
    "/api/product/{product}/info"
)
def api_product_info(
    product: str
):

    if product not in get_products():

        raise HTTPException(

            404,

            f"商品「{product}」不存在。"

        )

    return get_product_info(
        product
    )


@app.post(
    "/api/product/info"
)
def api_product_info_save(
    req: ProductInfoRequest
):

    try:

        product = req.product.strip()

        if product not in get_products():

            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_state()

        if (
            current["product"] == product
            and
            current["status"]
            in (
                "starting",
                "playing",
                "pause_requested",
                "paused"
            )
        ):

            raise RuntimeError(
                "直播正在运行，请先停止直播后再修改商品资料。"
            )

        if (
            get_tts_state().get(
                "product"
            ) == product
            and
            get_tts_state().get(
                "status"
            )
            in (
                "loading",
                "generating"
            )
        ):

            raise RuntimeError(
                "WAV 正在生成，请等待完成后再修改商品资料。"
            )

        save_product_info(
            product,
            req.model_dump()
        )

        clear_product_audio(
            product
        )

        return {

            "ok":
                True,

            "info":
                get_product_info(
                    product
                ),

            "message":
                "商品资料已保存，旧 WAV 已清理，请重新生成 WAV。"

        }

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# =========================================================
# 删除商品
# =========================================================

@app.delete(
    "/api/product/{product}"
)
def api_product_delete(
    product: str
):

    try:

        delete_product(
            product
        )

        return {

            "ok":
                True,

            "message":
                f"商品「{product}」及其话术、WAV 文件夹已删除。"

        }

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# =========================================================
# 商品话术
# =========================================================

@app.get(
    "/api/product/{product}"
)
def product_script(
    product: str
):

    script_file = (

        SCRIPT_DIR /
        product /
        "script.txt"

    )

    if not script_file.exists():

        raise HTTPException(

            404,

            "找不到该商品话术"

        )

    return {

        "product":
            product,

        "script":
            script_file.read_text(
                encoding="utf-8"
            ),

    }


# =========================================================
# 播放状态
# =========================================================

@app.get(
    "/api/status"
)
def status():

    return get_state()


# =========================================================
# AI状态
# =========================================================

@app.get(
    "/api/script/status"
)
def script_ai_status():

    return get_ai_state()


# =========================================================
# 话术版本
# =========================================================

@app.get(
    "/api/script/versions/{product}"
)
def script_versions(
    product: str
):

    if product not in get_products():

        raise HTTPException(

            404,

            f"商品「{product}」不存在。"

        )

    return {

        "product":
            product,

        "versions":
            get_script_versions(
                product
            )

    }


# =========================================================
# 选择话术版本
# =========================================================

@app.post(
    "/api/script/select"
)
def api_script_select(
    req: ProductRequest
):

    try:

        product = req.product.strip()

        if product not in get_products():

            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_state()

        if current["status"] in (

            "starting",
            "playing",
            "pause_requested",
            "paused"

        ):

            raise RuntimeError(
                "直播正在运行，请先停止直播。"
            )

        if get_tts_state()[
            "status"
        ] in (
            "loading",
            "generating"
        ):

            raise RuntimeError(
                "WAV 正在生成，请等待完成后再切换话术。"
            )

        select_script_version(

            product,

            req.style

        )

        return {

            "ok":
                True,

            "message":
                f"已切换到「{req.style}」，旧 WAV 已清理，请重新生成 WAV。"

        }

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# =========================================================
# AI生成当前风格
# =========================================================

@app.post(
    "/api/script/regenerate"
)
def api_script_regenerate(
    req: ProductRequest
):

    try:

        product = req.product.strip()

        if product not in get_products():

            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_state()

        if current["status"] in (

            "starting",
            "playing",
            "pause_requested",
            "paused"

        ):

            raise RuntimeError(
                "直播正在运行，请先停止直播。"
            )

        if get_tts_state()[
            "status"
        ] in (
            "loading",
            "generating"
        ):

            raise RuntimeError(
                "WAV 正在生成，请等待完成后再生成 AI 话术。"
            )

        style = (
            req.style
            or
            "高转化直播型"
        ).strip()

        if style not in SCRIPT_STYLES:

            raise RuntimeError(
                f"不支持的话术风格：{style}"
            )

        start_ai_script_generation(

            product,

            style,

            all_styles=False

        )

        return get_ai_state()

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# =========================================================
# AI生成全部
# =========================================================

@app.post(
    "/api/script/regenerate-all"
)
def api_script_regenerate_all(
    req: ProductRequest
):

    try:

        product = req.product.strip()

        if product not in get_products():

            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_state()

        if current["status"] in (

            "starting",
            "playing",
            "pause_requested",
            "paused"

        ):

            raise RuntimeError(
                "直播正在运行，请先停止直播。"
            )

        if get_tts_state()[
            "status"
        ] in (
            "loading",
            "generating"
        ):

            raise RuntimeError(
                "WAV 正在生成，请等待完成后再生成 AI 话术。"
            )

        start_ai_script_generation(

            product,

            all_styles=True

        )

        return get_ai_state()

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# =========================================================
# TTS状态
# =========================================================

@app.get(
    "/api/tts/status"
)
def tts_status():

    return get_tts_state()


# =========================================================
# 本地文件夹
# =========================================================

@app.post(
    "/api/local/open"
)
def api_local_open(
    req: LocalOpenRequest
):

    product = req.product.strip()

    kind = req.kind.strip().lower()


    if product not in get_products():

        raise HTTPException(

            404,

            f"商品「{product}」不存在。"

        )


    if kind == "script":

        target = SCRIPT_DIR / product

        label = "话术文件夹"

    elif kind == "wav":

        target = OUTPUT_DIR / product

        label = "WAV 文件夹"

    else:

        raise HTTPException(

            400,

            "不支持的本地目录类型。"

        )


    if not target.exists():

        raise HTTPException(

            404,

            f"{label}不存在，请先生成对应文件。"

        )


    try:

        if os.name == "nt":

            os.startfile(
                str(target)
            )

        elif (
            hasattr(
                os,
                "uname"
            )
            and
            os.uname().sysname
            == "Darwin"
        ):

            import subprocess

            subprocess.Popen(
                [
                    "open",
                    str(target)
                ]
            )

        else:

            import subprocess

            subprocess.Popen(
                [
                    "xdg-open",
                    str(target)
                ]
            )

        return {

            "ok":
                True,

            "message":
                f"已打开「{product}」的{label}。",

            "path":
                str(target)

        }

    except Exception as e:

        raise HTTPException(

            500,

            f"打开{label}失败：{e}"

        )


# =========================================================
# 开始直播
# =========================================================

@app.post(
    "/api/play/start"
)
def api_start(
    req: ProductRequest
):

    try:

        if (
            get_ai_state().get(
                "status"
            )
            ==
            "generating"
        ):

            raise RuntimeError(
                "AI 话术正在生成，请等待完成后再直播。"
            )

        if (
            get_tts_state().get(
                "status"
            )
            in (
                "loading",
                "generating"
            )
        ):

            raise RuntimeError(
                "WAV 正在生成，请等待完成后再直播。"
            )

        start_playback(
            req.product
        )

        return get_state()

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# =========================================================
# 暂停
# =========================================================

@app.post(
    "/api/play/pause"
)
def api_pause():

    return request_pause()


# =========================================================
# 继续
# =========================================================

@app.post(
    "/api/play/resume"
)
def api_resume():

    return resume_playback()


# =========================================================
# 停止
# =========================================================

@app.post(
    "/api/play/stop"
)
def api_stop():

    stop_playback()

    return get_state()


# =========================================================
# TTS生成
# =========================================================

@app.post(
    "/api/tts/generate"
)
def api_tts_generate(
    req: ProductRequest
):

    try:

        if req.product not in get_products():

            raise RuntimeError(
                f"商品「{req.product}」不存在。"
            )


        current = get_state()


        if current["status"] in (

            "playing",
            "pause_requested",
            "paused"

        ):

            raise RuntimeError(
                "直播正在运行，请先停止直播。"
            )


        current_ai = get_ai_state()


        if current_ai[
            "status"
        ] == "generating":

            raise RuntimeError(
                "AI 话术正在生成，请等待完成后再生成 WAV。"
            )


        start_tts_generation(
            req.product
        )


        return get_tts_state()

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# =========================================================
# 程序入口
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI直播控制台 V8")
    print("访问地址：http://127.0.0.1:8000")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )