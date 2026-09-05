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


# ============================================================
# 基础路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SCRIPT_DIR = BASE_DIR / "scripts"
OUTPUT_DIR = BASE_DIR / "output"

SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="AI直播控制台")


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


class AutoPlayRequest(BaseModel):
    product: str


class LiveCommentRequest(BaseModel):
    username: str = ""
    content: str = ""



# ============================================================
# AI Server 配置
#
# web_app.py  -> 8000
# ai_server.py -> 8001
# ============================================================

AI_SERVER_URL = os.getenv(
    "AI_SERVER_URL",
    "http://127.0.0.1:8001/comment"
)

AI_SERVER_TIMEOUT = 120


# ============================================================
# 播放器状态
# ============================================================

play_lock = threading.Lock()

play_thread: Optional[threading.Thread] = None

stop_event = threading.Event()

pause_event = threading.Event()

state_lock = threading.Lock()


player_state = {
    "status": "stopped",

    # 当前商品
    "product": "",

    # 普通话术播放位置
    "current_index": 0,
    "total": 0,

    # 当前话术信息
    "current_title": "",
    "current_text": "",
    "current_version": "",

    # 当前音频
    "current_audio": "",
    "current_audio_duration": 0,

    # 轮次
    "round_no": 0,

    # 播放进度
    "progress_percent": 0,

    # 直播时间
    "live_started_at": None,
    "live_stopped_at": None,

    # 提示
    "alert_level": "ok",
    "alert_message": "",

    # 当前状态消息
    "message": "等待开始直播",
}


# ============================================================
# 普通话术 TTS 状态
# ============================================================

tts_lock = threading.Lock()

tts_thread: Optional[threading.Thread] = None

tts_state_lock = threading.Lock()

tts_state = {
    "status": "idle",
    "product": "",
    "current_index": 0,
    "total": 0,
    "percent": 0,
    "current_tag": "",
    "current_version": "",
    "message": "等待生成",
    "error": "",
}


# ============================================================
# 普通话术 AI 生成状态
#
# 注意：
# 这里是原项目中用于生成普通直播话术的 AI 状态。
#
# 不要和下面的 live_ai_state 混用。
# ============================================================

ai_lock = threading.Lock()

ai_thread: Optional[threading.Thread] = None

ai_state_lock = threading.Lock()

ai_state = {
    "status": "idle",
    "product": "",
    "message": "等待生成",
    "error": "",
    "current": 0,
    "total": 0,
    "percent": 0,
    "current_style": "",
}


# ============================================================
# 直播间实时问答 AI
#
# 设计原则：
#
# 1. 观众问题进入 FIFO 队列
# 2. AI 在后台生成
# 3. AI 生成绝不能阻塞普通话术播放
# 4. AI WAV 生成完成后进入 ready 状态
# 5. 普通 WAV 播放结束时再播放 AI
# 6. 不允许 AI 中途打断普通 WAV
#
# 注意：
#
# live_ai_queue 同时保存：
#
# pending：
#     正在等待 AI/TTS 处理
#
# ready：
#     AI回答和WAV都已经准备好，等待普通话术结束后播放
#
# worker 只处理 pending，
# 绝不会重新处理 ready。
# ============================================================

live_ai_lock = threading.Lock()

live_ai_condition = threading.Condition(
    live_ai_lock
)

live_ai_queue = []

live_ai_thread: Optional[threading.Thread] = None

# 问题序号
live_ai_sequence = 0

# 当前直播会话编号
live_ai_session = 0


live_ai_state = {
    # idle / generating / ready / playing / error
    "status": "idle",

    # 当前等待处理/播放的 AI 数量
    "queue_size": 0,

    # 是否正在后台生成
    "generating": False,

    # 当前正在处理的问题
    "current_question": "",

    # 当前正在生成的回答
    "current_answer": "",

    # 最近一个问题
    "last_question": "",

    # 最近一个回答
    "last_answer": "",

    # 最近一次 AI Server 返回状态
    "last_status": "",

    # 页面显示消息
    "message": "等待观众提问",

    # 错误
    "error": "",
}


# ============================================================
# CosyVoice
# ============================================================

cosyvoice = None

model_lock = threading.Lock()


def load_cosyvoice():
    """
    延迟加载 CosyVoice。

    保持原项目的加载方式，
    避免启动 web_app.py 时立即占用大量显存/内存。
    """

    global cosyvoice

    if cosyvoice is not None:
        return cosyvoice

    with model_lock:

        if cosyvoice is not None:
            return cosyvoice

        cosyvoice = tts.AutoModel(
            model_dir=tts.MODEL_DIR
        )

    return cosyvoice


# ============================================================
# 播放器状态辅助函数
# ============================================================

def get_player_state():
    """
    返回播放器状态副本。
    """

    with state_lock:
        return dict(player_state)


def set_player_state(**kwargs):
    """
    修改播放器状态。
    """

    with state_lock:
        player_state.update(kwargs)


# ============================================================
# TTS 状态辅助函数
# ============================================================

def get_tts_state():
    """
    返回 TTS 状态副本。
    """

    with tts_state_lock:
        return dict(tts_state)


def set_tts_state(**kwargs):
    """
    修改 TTS 状态。
    """

    with tts_state_lock:
        tts_state.update(kwargs)


# ============================================================
# 普通话术 AI 状态辅助函数
# ============================================================

def get_ai_state():
    """
    返回普通话术 AI 状态副本。
    """

    with ai_state_lock:
        return dict(ai_state)


def set_ai_state(**kwargs):
    """
    修改普通话术 AI 状态。
    """

    with ai_state_lock:
        ai_state.update(kwargs)


# ============================================================
# 直播问答 AI 状态辅助函数
# ============================================================

def get_live_ai_state():
    """
    返回直播间 AI 问答状态。

    queue_size 每次实时计算，
    避免状态数字因为异常退出而不准确。
    """

    with live_ai_lock:

        state = dict(
            live_ai_state
        )

        state["queue_size"] = len(
            live_ai_queue
        )

        return state


def set_live_ai_state(**kwargs):
    """
    修改直播间 AI 状态。
    """

    with live_ai_lock:

        live_ai_state.update(
            kwargs
        )

        live_ai_state["queue_size"] = len(
            live_ai_queue
        )


# ============================================================
# 清空直播 AI 队列
# ============================================================

def clear_live_ai_queue():
    """
    清空当前直播的 AI 问答队列。

    同时：

    1. 删除已经生成但还没播放的 AI WAV
    2. 清空 FIFO 队列
    3. 重置问题序号
    4. 更新 session
    5. 重置 AI 状态

    session + 1 非常重要。

    如果 AI Server 正在生成回答，
    此时用户点击停止直播，
    后台生成可能还没结束。

    生成结束后通过 session 判断：

    如果已经不是当前直播，
    直接丢弃。

    不允许进入新一场直播。
    """

    global live_ai_sequence
    global live_ai_session

    with live_ai_condition:

        # ----------------------------------------------------
        # 删除已经生成好的 AI WAV
        # ----------------------------------------------------

        for task in live_ai_queue:

            wav_file = task.get(
                "wav_file",
                ""
            )

            if not wav_file:
                continue

            try:

                path = Path(
                    wav_file
                )

                if path.exists():
                    path.unlink()

            except Exception:
                pass

        # ----------------------------------------------------
        # 清空队列
        # ----------------------------------------------------

        live_ai_queue.clear()

        # ----------------------------------------------------
        # 新会话
        # ----------------------------------------------------

        live_ai_session += 1

        # ----------------------------------------------------
        # 序号重新开始
        # ----------------------------------------------------

        live_ai_sequence = 0

        # ----------------------------------------------------
        # 重置状态
        # ----------------------------------------------------

        live_ai_state.update({

            "status": "idle",

            "queue_size": 0,

            "generating": False,

            "current_question": "",

            "current_answer": "",

            "last_question": "",

            "last_answer": "",

            "last_status": "",

            "message": "等待观众提问",

            "error": "",
        })

        # ----------------------------------------------------
        # 唤醒后台线程
        # ----------------------------------------------------

        live_ai_condition.notify_all()


# ============================================================
# 添加观众问题到 FIFO 队列
# ============================================================

def enqueue_live_comment(
    username: str,
    question: str,
    product: str,
):
    """
    将观众问题加入 FIFO 队列。

    这里只负责入队。

    不在这里调用 DeepSeek。
    不在这里生成 WAV。
    不在这里播放声音。

    API 可以快速返回，
    不会因为 AI 生成而卡住网页请求。
    """

    global live_ai_sequence

    username = (
        username or ""
    ).strip()

    question = (
        question or ""
    ).strip()

    product = (
        product or ""
    ).strip()

    if not question:
        raise ValueError(
            "观众问题不能为空"
        )

    if not product:
        raise ValueError(
            "当前没有选择商品"
        )

    with live_ai_condition:

        live_ai_sequence += 1

        sequence = live_ai_sequence

        session_id = live_ai_session

        task = {

            "sequence": sequence,

            "session_id": session_id,

            "username": username,

            "product": product,

            "question": question,

            # AI 返回后填写
            "answer": "",

            # TTS 完成后填写
            "wav_file": "",

            # 是否已经可以播放
            "ready": False,

            # WAV时长
            "duration": 0,
        }

        # FIFO
        live_ai_queue.append(
            task
        )

        live_ai_state["status"] = (
            "generating"
        )

        live_ai_state["queue_size"] = (
            len(live_ai_queue)
        )

        live_ai_state["generating"] = True

        live_ai_state["current_question"] = (
            question
        )

        live_ai_state["current_answer"] = ""

        live_ai_state["message"] = (
            f"收到观众问题，正在后台处理：{question}"
        )

        live_ai_state["error"] = ""

        # 唤醒 AI 后台线程
        live_ai_condition.notify_all()

    # 确保 AI worker 已经启动
    start_live_ai_worker()

    return {
        "status": "queued",

        "sequence": sequence,

        "product": product,

        "question": question,
    }


# ============================================================
# 判断是否存在等待 AI 处理的任务
# ============================================================

def has_pending_live_ai_task():
    """
    判断队列中是否存在尚未完成 AI/TTS 的任务。

    ready=True 的任务不能被 worker 再次处理。
    """

    for task in live_ai_queue:

        if not task.get(
            "ready",
            False
        ):

            return True

    return False


# ============================================================
# 直播 AI：后台处理线程
# ============================================================

def live_ai_worker():
    """
    直播间 AI 后台处理线程。

    整个流程：

        观众提问
            ↓
        FIFO 队列
            ↓
        调用 ai_server.py
            ↓
        获取 AI 回答
            ↓
        CosyVoice 生成 WAV
            ↓
        标记 ready=True
            ↓
        等普通话术 WAV 播放结束
            ↓
        播放 AI

    最重要原则：

        AI 生成全过程都不能阻塞普通话术播放。

    注意：

        worker 只取 ready=False 的任务。

        已经生成完成的 ready 任务，
        绝对不会再次发送给 AI Server。
    """

    while True:

        # ----------------------------------------------------
        # 等待新的 pending 问题
        # ----------------------------------------------------

        with live_ai_condition:

            while (
                not has_pending_live_ai_task()
                or stop_event.is_set()
            ):

                live_ai_condition.wait(
                    timeout=0.5
                )

            # ------------------------------------------------
            # FIFO：
            # 找第一个尚未 ready 的任务
            # ------------------------------------------------

            task_index = None

            for index, candidate in enumerate(
                live_ai_queue
            ):

                if not candidate.get(
                    "ready",
                    False
                ):

                    task_index = index
                    break

            if task_index is None:
                continue

            # 从 pending 中取出
            task = live_ai_queue.pop(
                task_index
            )

            # 当前任务属于哪个直播 session
            task_session = task.get(
                "session_id",
                0
            )

            # 当前商品
            product = task.get(
                "product",
                ""
            )

            # 观众昵称
            username = task.get(
                "username",
                ""
            )

            # 问题
            question = task.get(
                "question",
                ""
            )

            # 问题序号
            sequence = task.get(
                "sequence",
                0
            )

        # ----------------------------------------------------
        # 任务已经从等待队列取出
        #
        # 此时不持有 live_ai_lock。
        #
        # 网络请求和 TTS 都不能锁住队列。
        # ----------------------------------------------------

        with live_ai_lock:

            current_session = (
                live_ai_session
            )

            live_ai_state["generating"] = True

            live_ai_state["status"] = (
                "generating"
            )

            live_ai_state["current_question"] = (
                question
            )

            live_ai_state["current_answer"] = ""

            live_ai_state["message"] = (
                f"正在后台回答：{question}"
            )

            live_ai_state["error"] = ""

            live_ai_state["queue_size"] = (
                len(live_ai_queue)
            )

        # ----------------------------------------------------
        # 检查任务是否已经过期
        # ----------------------------------------------------

        if (
            task_session != current_session
            or stop_event.is_set()
        ):

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "idle"
                )

                live_ai_state["message"] = (
                    "直播已停止，忽略旧问题"
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            continue

        # ----------------------------------------------------
        # 检查商品
        # ----------------------------------------------------

        if not product:

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "error"
                )

                live_ai_state["message"] = (
                    "当前没有商品，无法回答"
                )

                live_ai_state["error"] = (
                    "product is empty"
                )

            continue

        # ----------------------------------------------------
        # 调用 ai_server.py
        # ----------------------------------------------------

        try:

            response = requests.post(

                AI_SERVER_URL,

                json={
                    "username": username,
                    "content": question,
                    "product": product,
                },

                timeout=AI_SERVER_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

        except Exception as e:

            error_text = str(e)

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "error"
                )

                live_ai_state["message"] = (
                    "AI服务器请求失败"
                )

                live_ai_state["error"] = (
                    error_text
                )

                live_ai_state["last_status"] = (
                    "api_error"
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            print(
                "[LiveAI] AI Server 请求失败:",
                error_text
            )

            continue

        # ----------------------------------------------------
        # AI Server 返回状态
        # ----------------------------------------------------

        ai_status = data.get(
            "status",
            ""
        )

        answer = (
            data.get(
                "answer",
                ""
            )
            or ""
        ).strip()

        # ----------------------------------------------------
        # low_relevance
        #
        # 不生成 WAV
        # 不播放
        # 不打断普通话术
        # ----------------------------------------------------

        if ai_status == "low_relevance":

            distance = data.get(
                "distance"
            )

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "idle"
                )

                live_ai_state["last_question"] = (
                    question
                )

                live_ai_state["last_answer"] = (
                    answer
                )

                live_ai_state["last_status"] = (
                    "low_relevance"
                )

                live_ai_state["current_question"] = ""

                live_ai_state["current_answer"] = ""

                live_ai_state["message"] = (
                    "问题相关度较低，已跳过语音回答"
                )

                live_ai_state["error"] = ""

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            print(
                "[LiveAI] 跳过低相关问题:",
                question,
                "distance=",
                distance
            )

            # 绝对不能生成 WAV
            continue

        # ----------------------------------------------------
        # duplicate
        #
        # ai_server.py 负责 15 秒问题去重。
        # ----------------------------------------------------

        if ai_status == "duplicate":

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "idle"
                )

                live_ai_state["last_question"] = (
                    question
                )

                live_ai_state["last_answer"] = (
                    answer
                )

                live_ai_state["last_status"] = (
                    "duplicate"
                )

                live_ai_state["current_question"] = ""

                live_ai_state["current_answer"] = ""

                live_ai_state["message"] = (
                    "重复问题，已跳过"
                )

                live_ai_state["error"] = ""

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            print(
                "[LiveAI] 重复问题，跳过:",
                question
            )

            continue

        # ----------------------------------------------------
        # empty
        # ----------------------------------------------------

        if ai_status == "empty":

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "idle"
                )

                live_ai_state["last_question"] = (
                    question
                )

                live_ai_state["last_answer"] = ""

                live_ai_state["last_status"] = (
                    "empty"
                )

                live_ai_state["current_question"] = ""

                live_ai_state["current_answer"] = ""

                live_ai_state["message"] = (
                    "空问题，已跳过"
                )

                live_ai_state["error"] = ""

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            continue

        # ----------------------------------------------------
        # AI Server 其他错误
        # ----------------------------------------------------

        if ai_status != "success":

            error_text = data.get(
                "error",
                ""
            )

            if not error_text:

                error_text = (
                    f"AI Server status={ai_status}"
                )

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "error"
                )

                live_ai_state["last_question"] = (
                    question
                )

                live_ai_state["last_answer"] = (
                    answer
                )

                live_ai_state["last_status"] = (
                    ai_status
                )

                live_ai_state["current_question"] = ""

                live_ai_state["current_answer"] = ""

                live_ai_state["message"] = (
                    "AI回答失败"
                )

                live_ai_state["error"] = (
                    error_text
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            print(
                "[LiveAI] AI返回异常:",
                error_text
            )

            continue

        # ----------------------------------------------------
        # success 但没有回答
        # ----------------------------------------------------

        if not answer:

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "error"
                )

                live_ai_state["last_question"] = (
                    question
                )

                live_ai_state["last_answer"] = ""

                live_ai_state["last_status"] = (
                    "empty_answer"
                )

                live_ai_state["current_question"] = ""

                live_ai_state["current_answer"] = ""

                live_ai_state["message"] = (
                    "AI没有返回有效回答"
                )

                live_ai_state["error"] = (
                    "answer is empty"
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            continue

        # ----------------------------------------------------
        # AI 回答已经取得
        # ----------------------------------------------------

        with live_ai_lock:

            live_ai_state["current_answer"] = (
                answer
            )

            live_ai_state["last_question"] = (
                question
            )

            live_ai_state["last_answer"] = (
                answer
            )

            live_ai_state["last_status"] = (
                "success"
            )

            live_ai_state["message"] = (
                "AI回答生成完成，正在生成语音"
            )

            live_ai_state["error"] = ""

        print(
            "[LiveAI] AI回答:",
            answer
        )

        # ----------------------------------------------------
        # 再次检查直播 session
        # ----------------------------------------------------

        with live_ai_lock:

            current_session = (
                live_ai_session
            )

        if (
            task_session != current_session
            or stop_event.is_set()
        ):

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "idle"
                )

                live_ai_state["message"] = (
                    "直播已停止，丢弃旧回答"
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            continue

        # ----------------------------------------------------
        # 创建 AI WAV 目录
        # ----------------------------------------------------

        ai_output_dir = (
            OUTPUT_DIR
            / product
            / "_live_ai"
        )

        ai_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            ai_output_dir
            / f"AI_{sequence:06d}.wav"
        )

        try:

            if output_file.exists():
                output_file.unlink()

        except Exception:
            pass

        # ----------------------------------------------------
        # 加载 CosyVoice
        #
        # 注意：
        #
        # 这里不能再：
        #
        # with model_lock:
        #     load_cosyvoice()
        #
        # 因为 load_cosyvoice() 自己已经加锁。
        # ----------------------------------------------------

        try:

            model = load_cosyvoice()

        except Exception as e:

            error_text = str(e)

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "error"
                )

                live_ai_state["message"] = (
                    "CosyVoice加载失败"
                )

                live_ai_state["error"] = (
                    error_text
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            print(
                "[LiveAI] CosyVoice加载失败:",
                error_text
            )

            continue

        # ----------------------------------------------------
        # 生成 AI WAV
        #
        # 这里只生成。
        #
        # 不播放。
        # 不停止普通音频。
        # ----------------------------------------------------

        try:

            result_tts = tts.generate_tts(
                model,
                answer,
                str(output_file)
            )

            if result_tts is None:

                raise RuntimeError(
                    "tts.generate_tts 返回 None"
                )

        except Exception as e:

            error_text = str(e)

            try:

                if output_file.exists():
                    output_file.unlink()

            except Exception:
                pass

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "error"
                )

                live_ai_state["message"] = (
                    "AI语音生成失败"
                )

                live_ai_state["error"] = (
                    error_text
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            print(
                "[LiveAI] TTS生成失败:",
                error_text
            )

            continue

        # ----------------------------------------------------
        # 检查 WAV 是否真的生成
        # ----------------------------------------------------

        if not output_file.exists():

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "error"
                )

                live_ai_state["message"] = (
                    "AI语音文件没有生成"
                )

                live_ai_state["error"] = (
                    str(output_file)
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            continue

        # ----------------------------------------------------
        # TTS 完成以后再次检查 session
        # ----------------------------------------------------

        with live_ai_lock:

            current_session = (
                live_ai_session
            )

        if (
            task_session != current_session
            or stop_event.is_set()
        ):

            try:

                if output_file.exists():
                    output_file.unlink()

            except Exception:
                pass

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "idle"
                )

                live_ai_state["message"] = (
                    "直播已停止，丢弃旧AI语音"
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            continue

        # ----------------------------------------------------
        # WAV 有效性检查
        # ----------------------------------------------------

        try:

            with wave.open(
                str(output_file),
                "rb"
            ) as wf:

                frames = wf.getnframes()

                rate = wf.getframerate()

                if rate > 0:

                    duration = (
                        frames / float(rate)
                    )

                else:

                    duration = 0

        except Exception as e:

            error_text = str(e)

            try:

                if output_file.exists():
                    output_file.unlink()

            except Exception:
                pass

            with live_ai_lock:

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "error"
                )

                live_ai_state["message"] = (
                    "AI WAV文件无效"
                )

                live_ai_state["error"] = (
                    error_text
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

            print(
                "[LiveAI] WAV检查失败:",
                error_text
            )

            continue

        # ----------------------------------------------------
        # AI WAV 生成完成
        # ----------------------------------------------------

        task["answer"] = answer

        task["wav_file"] = str(
            output_file
        )

        task["duration"] = duration

        task["ready"] = True

        # ----------------------------------------------------
        # 重新加入 ready 队列
        # ----------------------------------------------------

        with live_ai_condition:

            # 再次确认 session
            if (
                task_session != live_ai_session
                or stop_event.is_set()
            ):

                try:

                    if output_file.exists():
                        output_file.unlink()

                except Exception:
                    pass

                live_ai_state["generating"] = False

                live_ai_state["status"] = (
                    "idle"
                )

                live_ai_state["message"] = (
                    "直播已停止，丢弃AI语音"
                )

                live_ai_state["queue_size"] = (
                    len(live_ai_queue)
                )

                continue

            # 添加到 ready 队列
            live_ai_queue.append(
                task
            )

            # sequence 决定 FIFO 顺序
            live_ai_queue.sort(
                key=lambda item: item.get(
                    "sequence",
                    0
                )
            )

            live_ai_state["generating"] = False

            live_ai_state["status"] = (
                "ready"
            )

            live_ai_state["current_question"] = ""

            live_ai_state["current_answer"] = ""

            live_ai_state["last_question"] = (
                question
            )

            live_ai_state["last_answer"] = (
                answer
            )

            live_ai_state["last_status"] = (
                "success"
            )

            live_ai_state["message"] = (
                "AI回答已准备完成，等待普通话术结束后播放"
            )

            live_ai_state["error"] = ""

            live_ai_state["queue_size"] = (
                len(live_ai_queue)
            )

            live_ai_condition.notify_all()

        print(
            "[LiveAI] AI语音准备完成:",
            output_file,
            "duration=",
            round(duration, 2),
            "秒"
        )


# ============================================================
# 启动直播 AI 后台线程
# ============================================================

def start_live_ai_worker():
    """
    确保直播 AI 后台线程只启动一次。
    """

    global live_ai_thread

    with live_ai_lock:

        if (
            live_ai_thread is not None
            and live_ai_thread.is_alive()
        ):

            return

        live_ai_thread = threading.Thread(
            target=live_ai_worker,
            name="LiveAIWorker",
            daemon=True,
        )

        live_ai_thread.start()

    print(
        "[LiveAI] 后台AI线程已启动"
    )


# ============================================================
# 播放一个已经生成好的 AI 回答
# ============================================================

def pop_ready_live_ai():
    """
    从 FIFO 队列中取出一个已经生成好的 AI 回答。

    注意：

    这里不会等待 AI 生成。

    如果当前没有 ready 的 AI，
    立即返回 None。

    因此普通话术永远不会因为 AI 没准备好而等待。
    """

    stale_files = []

    selected_task = None

    with live_ai_condition:

        if not live_ai_queue:
            return None

        current_session = (
            live_ai_session
        )

        current_product = (
            player_state.get(
                "product",
                ""
            )
        )

        # ----------------------------------------------------
        # 找第一个有效 ready AI
        # ----------------------------------------------------

        for index, task in enumerate(
            live_ai_queue
        ):

            # 只处理已经完成 TTS 的任务
            if not task.get(
                "ready",
                False
            ):

                continue

            # session 不一致
            if task.get(
                "session_id"
            ) != current_session:

                stale_files.append(
                    task.get(
                        "wav_file",
                        ""
                    )
                )

                continue

            # 商品不一致
            if (
                task.get(
                    "product",
                    ""
                )
                != current_product
            ):

                stale_files.append(
                    task.get(
                        "wav_file",
                        ""
                    )
                )

                continue

            # WAV 不存在
            wav_file = task.get(
                "wav_file",
                ""
            )

            if not wav_file:
                continue

            if not Path(
                wav_file
            ).exists():

                stale_files.append(
                    wav_file
                )

                continue

            # 找到了
            selected_task = (
                live_ai_queue.pop(
                    index
                )
            )

            break

        live_ai_state["queue_size"] = (
            len(live_ai_queue)
        )

    # --------------------------------------------------------
    # 清理失效 WAV
    # --------------------------------------------------------

    for wav_file in stale_files:

        if not wav_file:
            continue

        try:

            path = Path(
                wav_file
            )

            if path.exists():
                path.unlink()

        except Exception:
            pass

    return selected_task


# ============================================================
# 播放 AI 回答
# ============================================================

def play_live_ai_task(task):
    """
    播放一个 AI 回答。

    只有普通 WAV 播放结束以后，
    playback_worker 才会调用这个函数。

    因此不会发生：

        普通话术播放到一半
        ↓
        AI突然插入
    """

    if not task:
        return False

    wav_file = task.get(
        "wav_file",
        ""
    )

    answer = task.get(
        "answer",
        ""
    )

    sequence = task.get(
        "sequence",
        0
    )

    if not wav_file:
        return False

    wav_path = Path(
        wav_file
    )

    if not wav_path.exists():
        return False

    if stop_event.is_set():
        return False

    # --------------------------------------------------------
    # 设置状态
    # --------------------------------------------------------

    set_live_ai_state(
        status="playing",
        generating=False,
        current_question=task.get(
            "question",
            ""
        ),
        current_answer=answer,
        message="正在播放AI回答",
        error="",
    )

    set_player_state(
        message="正在播放AI回答",
        alert_level="ok",
        alert_message="",
    )

    print(
        "[LiveAI] 开始播放 AI:",
        sequence,
        answer
    )

    # --------------------------------------------------------
    # 播放 AI WAV
    #
    # 不调用 stop_audio_device()
    # --------------------------------------------------------

    try:

        tts.play_wav(
            wav_file,
            1.0
        )

    except Exception as e:

        error_text = str(e)

        set_live_ai_state(
            status="error",
            generating=False,
            message="AI回答播放失败",
            error=error_text,
        )

        print(
            "[LiveAI] AI播放失败:",
            error_text
        )

        return False

    finally:

        try:

            if wav_path.exists():
                wav_path.unlink()

        except Exception:
            pass

    # --------------------------------------------------------
    # AI 播放结束
    # --------------------------------------------------------

    if stop_event.is_set():

        set_live_ai_state(
            status="idle",
            generating=False,
            current_question="",
            current_answer="",
            message="直播已停止",
            error="",
        )

        return False

    set_live_ai_state(
        status="idle",
        generating=False,
        current_question="",
        current_answer="",
        message="AI回答完成，继续普通话术",
        error="",
    )

    set_player_state(
        message="AI回答完成，继续普通话术",
    )

    print(
        "[LiveAI] AI回答播放完成:",
        sequence
    )

    return True


# ============================================================
# 兼容不同 playback_versions 结构
# ============================================================

def normalize_playback_items(playback_versions):
    """
    将原项目可能存在的不同 playback_versions 结构
    统一转换成：

        [
            {
                "title": ...,
                "text": ...,
                "version": ...,
                "wav": ...,
                "pause": ...
            }
        ]

    支持：

    结构 A：

        [
            {
                "title": "...",
                "text": "...",
                "version": "...",
                "wav": "..."
            }
        ]

    结构 B：

        [
            {
                "style": "...",
                "items": [
                    {
                        "title": "...",
                        "text": "...",
                        "wav": "..."
                    }
                ]
            }
        ]

    这样可以兼容原 GitHub 版本。
    """

    if not playback_versions:
        return []

    normalized = []

    # --------------------------------------------------------
    # 情况 A：
    # 已经是普通 item 列表
    # --------------------------------------------------------

    if isinstance(
        playback_versions,
        list
    ):

        for entry in playback_versions:

            if not isinstance(
                entry,
                dict
            ):

                continue

            # ------------------------------------------------
            # 情况 B：
            # style + items
            # ------------------------------------------------

            if isinstance(
                entry.get("items"),
                list
            ):

                style = entry.get(
                    "style",
                    ""
                )

                for item in entry.get(
                    "items",
                    []
                ):

                    if not isinstance(
                        item,
                        dict
                    ):

                        continue

                    data = dict(
                        item
                    )

                    if not data.get(
                        "version",
                        ""
                    ):

                        data["version"] = (
                            style
                        )

                    normalized.append(
                        data
                    )

                continue

            # ------------------------------------------------
            # 普通 item
            # ------------------------------------------------

            normalized.append(
                dict(entry)
            )

    return normalized


# ============================================================
# 播放器：核心播放线程
# ============================================================

def playback_worker(
    product: str,
    playback_versions
):
    """
    普通直播话术播放线程。

    核心逻辑：

        A.wav
          ↓
        播放完成
          ↓
        检查 AI
          ↓
        AI已准备好？
        ├── 是 → 播放 AI
        └── 否 → 不等待，继续 B.wav
          ↓
        B.wav
          ↓
        ...

    AI绝对不允许：

        A.wav 播放到一半
             ↓
        AI生成完成
             ↓
        停止 A
             ↓
        播放 AI

    AI只能在普通 WAV 完整播放结束以后播放。
    """

    # --------------------------------------------------------
    # 统一 playback_versions
    # --------------------------------------------------------

    playback_items = (
        normalize_playback_items(
            playback_versions
        )
    )

    index = 0

    total = len(
        playback_items
    )

    round_no = 0

    # --------------------------------------------------------
    # 初始状态
    # --------------------------------------------------------

    set_player_state(
        status="playing",
        product=product,
        current_index=0,
        total=total,
        current_title="",
        current_text="",
        current_version="",
        current_audio="",
        current_audio_duration=0,
        round_no=0,
        progress_percent=0,
        alert_level="ok",
        alert_message="",
        message="开始播放普通话术",
    )

    # --------------------------------------------------------
    # 确保 Live AI Worker 已启动
    # --------------------------------------------------------

    start_live_ai_worker()

    # ========================================================
    # 主循环
    # ========================================================

    while not stop_event.is_set():

        # ----------------------------------------------------
        # 没有普通话术
        # ----------------------------------------------------

        if not playback_items:

            set_player_state(
                status="error",
                message="没有可播放的话术",
                alert_level="error",
                alert_message="没有可播放的话术",
            )

            break

        # ----------------------------------------------------
        # 完成一轮
        # ----------------------------------------------------

        if index >= total:

            index = 0

            round_no += 1

            set_player_state(
                round_no=round_no,
                current_index=0,
                progress_percent=0,
                message=f"第 {round_no} 轮开始",
            )

        # ----------------------------------------------------
        # 获取当前话术
        # ----------------------------------------------------

        item = playback_items[index]

        title = item.get(
            "title",
            ""
        )

        text = item.get(
            "text",
            ""
        )

        version = item.get(
            "version",
            ""
        )

        wav_file = item.get(
            "wav",
            ""
        )

        if not wav_file:

            wav_file = item.get(
                "audio",
                ""
            )

        # ----------------------------------------------------
        # pause
        # ----------------------------------------------------

        try:

            pause_time = float(
                item.get(
                    "pause",
                    0.7
                )
            )

        except Exception:

            pause_time = 0.7

        if pause_time < 0:
            pause_time = 0

        # ----------------------------------------------------
        # WAV 检查
        # ----------------------------------------------------

        if not wav_file:

            set_player_state(
                status="error",
                current_index=index,
                current_title=title,
                current_text=text,
                current_version=version,
                current_audio="",
                message=(
                    f"话术没有音频文件：{title}"
                ),
                alert_level="error",
                alert_message=(
                    f"话术没有音频文件：{title}"
                ),
            )

            index += 1

            continue

        wav_path = Path(
            wav_file
        )

        if not wav_path.exists():

            set_player_state(
                status="error",
                current_index=index,
                current_title=title,
                current_text=text,
                current_version=version,
                current_audio=str(
                    wav_path
                ),
                message=(
                    f"音频文件不存在：{wav_path}"
                ),
                alert_level="error",
                alert_message=(
                    f"音频文件不存在：{wav_path}"
                ),
            )

            index += 1

            continue

        # ----------------------------------------------------
        # 获取 WAV 时长
        # ----------------------------------------------------

        duration = 0

        try:

            with wave.open(
                str(wav_path),
                "rb"
            ) as wf:

                frames = wf.getnframes()

                rate = wf.getframerate()

                if rate > 0:

                    duration = (
                        frames / float(rate)
                    )

        except Exception:

            duration = 0

        # ----------------------------------------------------
        # 当前话术状态
        # ----------------------------------------------------

        percent = 0

        if total > 0:

            percent = (
                index / float(total)
            ) * 100

        set_player_state(
            status="playing",
            product=product,
            current_index=index,
            total=total,
            current_title=title,
            current_text=text,
            current_version=version,
            current_audio=str(
                wav_path
            ),
            current_audio_duration=duration,
            round_no=round_no,
            progress_percent=percent,
            alert_level="ok",
            alert_message="",
            message=(
                f"正在播放：{title}"
            ),
        )

        print(
            "[PLAY]",
            f"round={round_no}",
            f"index={index + 1}/{total}",
            f"title={title}",
            f"wav={wav_path}",
        )

        # ====================================================
        # 播放普通 WAV
        #
        # 关键：
        #
        # AI 不会在这里执行。
        #
        # tts.play_wav() 返回之前，
        # AI绝对不会插入。
        # ====================================================

        try:

            tts.play_wav(
                str(wav_path),
                1.0
            )

        except Exception as audio_error:

            if stop_event.is_set():
                break

            error_text = str(
                audio_error
            )

            set_player_state(
                status="error",
                message=(
                    f"音频播放失败：{error_text}"
                ),
                alert_level="error",
                alert_message=error_text,
            )

            print(
                "[PLAY] 音频播放失败:",
                error_text
            )

            break

        # ====================================================
        # ★★★ 普通 WAV 到这里才真正结束 ★★★
        # ====================================================

        if stop_event.is_set():
            break

        # ----------------------------------------------------
        # 暂停
        # ----------------------------------------------------

        if pause_event.is_set():

            set_player_state(
                status="paused",
                message="已暂停，等待继续",
            )

            while (
                pause_event.is_set()
                and not stop_event.is_set()
            ):

                time.sleep(
                    0.1
                )

            if stop_event.is_set():
                break

            set_player_state(
                status="playing",
                message="继续播放",
            )

        # ====================================================
        # ★★★ AI 插入点 ★★★
        #
        # 只检查，不等待。
        # ====================================================

        ai_task = pop_ready_live_ai()

        if ai_task is not None:

            ai_played = (
                play_live_ai_task(
                    ai_task
                )
            )

            if stop_event.is_set():
                break

            if ai_played:

                set_player_state(
                    status="playing",
                    product=product,
                    current_index=index,
                    total=total,
                    current_title=title,
                    current_text=text,
                    current_version=version,
                    current_audio=str(
                        wav_path
                    ),
                    current_audio_duration=duration,
                    round_no=round_no,
                    progress_percent=(
                        index / float(total) * 100
                        if total > 0
                        else 0
                    ),
                    alert_level="ok",
                    alert_message="",
                    message=(
                        "AI回答完成，继续普通话术"
                    ),
                )

                print(
                    "[PLAY] AI回答结束，继续普通话术"
                )

        # ====================================================
        # 普通话术句间停顿
        # ====================================================

        if pause_time > 0:

            if stop_event.wait(
                pause_time
            ):

                break

        # ----------------------------------------------------
        # 下一条
        # ----------------------------------------------------

        index += 1

        next_percent = 0

        if total > 0:

            next_percent = (
                index / float(total)
            ) * 100

        set_player_state(
            current_index=index,
            progress_percent=next_percent,
        )

    # ========================================================
    # 播放线程退出
    # ========================================================

    if stop_event.is_set():

        set_player_state(
            status="stopped",
            message="直播已停止",
            progress_percent=0,
        )

    else:

        current = get_player_state()

        if current.get(
            "status"
        ) != "error":

            set_player_state(
                status="stopped",
                message="播放结束",
            )

    print(
        "[PLAY] playback_worker 已退出"
    )


# ============================================================
# 开始播放
# ============================================================

def start_playback(product):
    """
    开始直播普通话术播放。

    同时初始化本场直播的 Live AI。
    """

    global play_thread

    product = (
        product or ""
    ).strip()

    if not product:

        raise ValueError(
            "请选择商品"
        )

    with play_lock:

        # ----------------------------------------------------
        # 如果已有播放线程
        # ----------------------------------------------------

        if (
            play_thread is not None
            and play_thread.is_alive()
        ):

            current_state = (
                get_player_state()
            )

            current_product = (
                current_state.get(
                    "product",
                    ""
                )
            )

            if current_product == product:

                return {
                    "status": "already_playing",
                    "product": product,
                }

            # ------------------------------------------------
            # 切换商品
            # ------------------------------------------------

            print(
                "[PLAY] 切换商品:",
                current_product,
                "->",
                product
            )

            stop_event.set()

            pause_event.clear()

            try:

                stop_audio_device()

            except Exception:
                pass

            old_thread = play_thread

            if (
                old_thread is not None
                and old_thread.is_alive()
            ):

                old_thread.join(
                    timeout=3
                )

            play_thread = None

        # ----------------------------------------------------
        # 清理上一场直播
        # ----------------------------------------------------

        clear_live_ai_queue()

        # ----------------------------------------------------
        # 清除停止和暂停状态
        # ----------------------------------------------------

        stop_event.clear()

        pause_event.clear()

        # ----------------------------------------------------
        # 启动 Live AI Worker
        # ----------------------------------------------------

        start_live_ai_worker()

        # ----------------------------------------------------
        # 获取当前商品播放版本
        # ----------------------------------------------------

        playback_versions = (
            get_playback_versions(
                product
            )
        )

        if not playback_versions:

            set_player_state(
                status="error",
                product=product,
                message=(
                    f"商品「{product}」没有可播放的话术"
                ),
                alert_level="error",
                alert_message=(
                    f"商品「{product}」没有可播放的话术"
                ),
            )

            raise ValueError(
                f"商品「{product}」没有可播放的话术"
            )

        # ----------------------------------------------------
        # 统一计算实际播放数量
        # ----------------------------------------------------

        playback_items = (
            normalize_playback_items(
                playback_versions
            )
        )

        if not playback_items:

            raise ValueError(
                f"商品「{product}」没有可播放的话术"
            )

        # ----------------------------------------------------
        # 开始时间
        # ----------------------------------------------------

        started_at = time.time()

        # ----------------------------------------------------
        # 初始化播放器状态
        # ----------------------------------------------------

        set_player_state(
            status="starting",
            product=product,
            current_index=0,
            total=len(
                playback_items
            ),
            current_title="",
            current_text="",
            current_version="",
            current_audio="",
            current_audio_duration=0,
            round_no=0,
            progress_percent=0,
            live_started_at=started_at,
            live_stopped_at=None,
            alert_level="ok",
            alert_message="",
            message="正在启动直播播放",
        )

        # ----------------------------------------------------
        # 创建播放线程
        # ----------------------------------------------------

        play_thread = threading.Thread(
            target=playback_worker,
            args=(
                product,
                playback_versions,
            ),
            name="PlaybackWorker",
            daemon=True,
        )

        play_thread.start()

    print(
        "[PLAY] 播放线程已启动:",
        product
    )

    return {
        "status": "started",
        "product": product,
        "total": len(
            playback_items
        ),
    }


# ============================================================
# 停止播放
# ============================================================

def stop_playback():
    """
    停止直播。

    stop_audio_device() 只在用户主动停止直播时调用。

    AI 正常插入播放时绝对不会调用它。
    """

    global play_thread

    print(
        "[PLAY] 收到停止直播请求"
    )

    # --------------------------------------------------------
    # 立即设置停止
    # --------------------------------------------------------

    stop_event.set()

    pause_event.clear()

    # --------------------------------------------------------
    # 清空 Live AI
    # --------------------------------------------------------

    clear_live_ai_queue()

    # --------------------------------------------------------
    # 主动停止当前音频
    # --------------------------------------------------------

    try:

        stop_audio_device()

    except Exception as e:

        print(
            "[PLAY] 停止音频失败:",
            e
        )

    # --------------------------------------------------------
    # 等待播放线程退出
    # --------------------------------------------------------

    current_thread = play_thread

    if (
        current_thread is not None
        and current_thread.is_alive()
        and current_thread is not threading.current_thread()
    ):

        current_thread.join(
            timeout=3
        )

    # --------------------------------------------------------
    # 清空线程引用
    # --------------------------------------------------------

    with play_lock:

        if (
            play_thread is current_thread
        ):

            play_thread = None

    # --------------------------------------------------------
    # 更新状态
    # --------------------------------------------------------

    stopped_at = time.time()

    current_state = (
        get_player_state()
    )

    product = (
        current_state.get(
            "product",
            ""
        )
    )

    set_player_state(
        status="stopped",
        product=product,
        live_stopped_at=stopped_at,
        current_audio="",
        current_audio_duration=0,
        alert_level="ok",
        alert_message="",
        message="直播已停止",
    )

    print(
        "[PLAY] 直播已停止"
    )

    return {
        "status": "stopped",
        "product": product,
    }


# ============================================================
# 暂停播放
# ============================================================

def pause_playback():
    """
    请求暂停。

    当前普通 WAV 不会立即切断。

    当前这一句话播放完成以后，
    playback_worker 才会进入暂停。

    AI也不会因为暂停请求被强制切断。
    """

    current_state = (
        get_player_state()
    )

    status = current_state.get(
        "status",
        "stopped"
    )

    if status not in (
        "playing",
        "starting",
    ):

        return {
            "status": "ignored",
            "message": "当前没有正在播放的直播",
        }

    pause_event.set()

    set_player_state(
        status="playing",
        message="已请求暂停，当前语音播放完成后暂停",
    )

    print(
        "[PLAY] 已请求暂停"
    )

    return {
        "status": "pause_requested",
        "message": "当前语音播放完成后暂停",
    }


# ============================================================
# 恢复播放
# ============================================================

def resume_playback():
    """
    恢复直播播放。
    """

    current_state = (
        get_player_state()
    )

    status = current_state.get(
        "status",
        "stopped"
    )

    if status not in (
        "paused",
        "playing",
    ):

        return {
            "status": "ignored",
            "message": "当前没有暂停的直播",
        }

    pause_event.clear()

    set_player_state(
        status="playing",
        message="继续播放直播",
    )

    print(
        "[PLAY] 直播已恢复"
    )

    return {
        "status": "resumed",
        "message": "直播已恢复",
    }


# ============================================================
# 直播间 AI 观众提问 API
# ============================================================

class LiveCommentRequest(BaseModel):
    """
    直播间观众提问请求。

    username：
        观众昵称，可为空。

    content：
        观众问题。

    product：
        不需要前端传。

        后端自动读取当前正在播放的商品。
    """

    username: str = ""

    content: str


# ============================================================
# POST /api/live/comment
# ============================================================

@app.post("/api/live/comment")
def api_live_comment(
    request: LiveCommentRequest
):
    """
    接收直播间观众问题。

    API收到问题以后立即返回。

    不等待：

        DeepSeek
        CosyVoice
        WAV生成
        WAV播放
    """

    username = (
        request.username or ""
    ).strip()

    question = (
        request.content or ""
    ).strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="观众问题不能为空",
        )

    # --------------------------------------------------------
    # 获取当前播放器状态
    # --------------------------------------------------------

    current_state = (
        get_player_state()
    )

    player_status = (
        current_state.get(
            "status",
            "stopped"
        )
    )

    if player_status in (
        "stopped",
        "error",
    ):

        raise HTTPException(
            status_code=400,
            detail="当前没有正在进行的直播",
        )

    # --------------------------------------------------------
    # ★ 当前商品以播放器状态为准
    # --------------------------------------------------------

    product = (
        current_state.get(
            "product",
            ""
        )
        or ""
    ).strip()

    if not product:

        raise HTTPException(
            status_code=400,
            detail="当前没有选择商品",
        )

    # --------------------------------------------------------
    # 加入 FIFO
    # --------------------------------------------------------

    try:

        result = enqueue_live_comment(
            username=username,
            question=question,
            product=product,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        print(
            "[LiveAI] 加入问题队列失败:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"加入AI队列失败：{e}"
            ),
        )

    # --------------------------------------------------------
    # 只更新 message
    #
    # 不改变 playing 状态。
    # --------------------------------------------------------

    try:

        current_player_status = (
            get_player_state().get(
                "status",
                ""
            )
        )

        if current_player_status in (
            "playing",
            "starting",
        ):

            set_player_state(
                message=(
                    "收到观众问题，AI正在后台处理"
                )
            )

    except Exception:

        pass

    return {
        "status": "queued",

        "sequence": result.get(
            "sequence",
            0
        ),

        "product": product,

        "question": question,

        "message": (
            "已收到观众问题，AI将在后台处理"
        ),
    }


# ============================================================
# GET /api/live/comment/status
# ============================================================

@app.get("/api/live/comment/status")
def api_live_comment_status():
    """
    获取直播间 AI 问答状态。
    """

    state = (
        get_live_ai_state()
    )

    return {
        "status": "ok",
        "data": state,
    }


# ============================================================
# GET /api/live/comment/queue
# ============================================================

@app.get("/api/live/comment/queue")
def api_live_comment_queue():
    """
    获取当前 AI 待处理/待播放队列。
    """

    result = []

    with live_ai_lock:

        for task in live_ai_queue:

            result.append({

                "sequence": task.get(
                    "sequence",
                    0
                ),

                "username": task.get(
                    "username",
                    ""
                ),

                "product": task.get(
                    "product",
                    ""
                ),

                "question": task.get(
                    "question",
                    ""
                ),

                "answer": task.get(
                    "answer",
                    ""
                ),

                "ready": bool(
                    task.get(
                        "ready",
                        False
                    )
                ),

                "wav_file": bool(
                    task.get(
                        "wav_file",
                        ""
                    )
                ),

                "session_id": task.get(
                    "session_id",
                    0
                ),
            })

    return {
        "status": "ok",

        "count": len(result),

        "items": result,
    }


# ============================================================
# 系统总状态
#
# ★★★ 这里是唯一一个 /api/status ★★★
#
# 必须保持原 GitHub 前端需要的顶层结构。
#
# 不要改成：
#
# {
#     "player": ...,
#     "tts": ...,
#     "ai": ...,
#     "live_ai": ...
# }
# ============================================================

@app.get("/api/status")
def api_status():

    return get_player_state()


# ============================================================
# Live AI 状态接口
# ============================================================

@app.get("/api/live/ai/status")
def api_live_ai_status():

    return get_live_ai_state()


# ============================================================
# Live AI 测试接口
# ============================================================

@app.post("/api/live/ai/test")
def api_live_ai_test():
    """
    手动测试 Live AI。

    测试时自动使用当前播放器正在播放的商品。
    """

    current_player = (
        get_player_state()
    )

    if current_player.get(
        "status"
    ) in (
        "stopped",
        "error",
    ):

        raise HTTPException(
            status_code=400,
            detail="当前没有正在进行的直播"
        )

    product = (
        current_player.get(
            "product",
            ""
        )
        or ""
    ).strip()

    if not product:

        raise HTTPException(
            status_code=400,
            detail="当前没有选择商品"
        )

    result = enqueue_live_comment(
        username="测试观众",
        question="这个商品有什么特点？",
        product=product,
    )

    return result

# =========================================================
# 第7部分：商品 / 话术 / TTS / 本地文件 API
# =========================================================


# =========================================================
# 商品列表
# =========================================================

@app.get("/api/products")
def products():
    return {
        "products": get_products()
    }


# =========================================================
# 创建商品
# =========================================================

@app.post("/api/product/create")
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
            "ok": True,
            "product": product,
            "message": f"商品「{product}」添加成功。",
            "info": get_product_info(product)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 获取商品资料
# =========================================================

@app.get("/api/product/{product}/info")
def api_product_info(
    product: str
):
    try:
        product = validate_product_name(product)

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        return get_product_info(product)

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )


# =========================================================
# 保存商品资料
# =========================================================

@app.post("/api/product/info")
def api_product_info_save(
    req: ProductInfoRequest
):
    try:

        product = validate_product_name(
            req.product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_player_state()

        # 直播过程中禁止修改商品资料
        if (
            current.get("product") == product
            and
            current.get("status")
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

        # WAV生成过程中禁止修改
        tts_current = get_tts_state()

        if (
            tts_current.get("product") == product
            and
            tts_current.get("status")
            in (
                "loading",
                "generating"
            )
        ):
            raise RuntimeError(
                "WAV 正在生成，请等待完成后再修改商品资料。"
            )

        # 保存商品资料
        save_product_info(
            product,
            req.model_dump()
        )

        # 商品资料发生变化后，
        # 原来的 WAV 必须清理
        clear_product_audio(
            product
        )

        return {
            "ok": True,
            "info": get_product_info(product),
            "message":
                "商品资料已保存，旧 WAV 已清理，请重新生成 WAV。"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 删除商品
# =========================================================

@app.delete("/api/product/{product}")
def api_product_delete(
    product: str
):
    try:

        product = validate_product_name(
            product
        )

        delete_product(
            product
        )

        return {
            "ok": True,
            "message":
                f"商品「{product}」及其话术、WAV 文件夹已删除。"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 获取商品当前话术
# =========================================================

@app.get("/api/product/{product}")
def product_script(
    product: str
):
    try:
        product = validate_product_name(
            product
        )

        script_file = (
            SCRIPT_DIR
            / product
            / "script.txt"
        )

        if not script_file.exists():
            raise HTTPException(
                status_code=404,
                detail="找不到该商品话术"
            )

        return {
            "product": product,
            "script":
                script_file.read_text(
                    encoding="utf-8"
                )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 话术 AI 状态
# =========================================================

@app.get("/api/script/status")
def script_ai_status():
    return get_ai_state()


# =========================================================
# 获取商品所有话术版本
# =========================================================

@app.get("/api/script/versions/{product}")
def script_versions(
    product: str
):
    try:

        product = validate_product_name(
            product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        return {
            "product": product,
            "versions":
                get_script_versions(
                    product
                )
        }

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )


# =========================================================
# 切换话术版本
# =========================================================

@app.post("/api/script/select")
def api_script_select(
    req: ProductRequest
):
    try:

        product = validate_product_name(
            req.product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_player_state()

        # 直播过程中不能切换话术
        if current.get("status") in (
            "starting",
            "playing",
            "pause_requested",
            "paused"
        ):
            raise RuntimeError(
                "直播正在运行，请先停止直播。"
            )

        # WAV生成过程中不能切换
        tts_current = get_tts_state()

        if tts_current.get("status") in (
            "loading",
            "generating"
        ):
            raise RuntimeError(
                "WAV 正在生成，请等待完成后再切换话术。"
            )

        style = (
            req.style
            or "高转化直播型"
        ).strip()

        if style not in SCRIPT_STYLES:
            raise RuntimeError(
                f"不支持的话术风格：{style}"
            )

        select_script_version(
            product,
            style
        )

        return {
            "ok": True,
            "message":
                f"已切换到「{style}」，旧 WAV 已清理，请重新生成 WAV。"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# AI重新生成当前风格话术
# =========================================================

@app.post("/api/script/regenerate")
def api_script_regenerate(
    req: ProductRequest
):
    try:

        product = validate_product_name(
            req.product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_player_state()

        if current.get("status") in (
            "starting",
            "playing",
            "pause_requested",
            "paused"
        ):
            raise RuntimeError(
                "直播正在运行，请先停止直播。"
            )

        tts_current = get_tts_state()

        if tts_current.get("status") in (
            "loading",
            "generating"
        ):
            raise RuntimeError(
                "WAV 正在生成，请等待完成后再生成 AI 话术。"
            )

        style = (
            req.style
            or "高转化直播型"
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
            status_code=400,
            detail=str(e)
        )


# =========================================================
# AI生成全部话术版本
# =========================================================

@app.post("/api/script/regenerate-all")
def api_script_regenerate_all(
    req: ProductRequest
):
    try:

        product = validate_product_name(
            req.product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_player_state()

        if current.get("status") in (
            "starting",
            "playing",
            "pause_requested",
            "paused"
        ):
            raise RuntimeError(
                "直播正在运行，请先停止直播。"
            )

        tts_current = get_tts_state()

        if tts_current.get("status") in (
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
            status_code=400,
            detail=str(e)
        )


# =========================================================
# TTS状态
# =========================================================

@app.get("/api/tts/status")
def tts_status():
    return get_tts_state()


# =========================================================
# 打开本地话术 / WAV 文件夹
# =========================================================

@app.post("/api/local/open")
def api_local_open(
    req: LocalOpenRequest
):
    try:

        product = validate_product_name(
            req.product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        kind = (
            req.kind
            or ""
        ).strip().lower()

        if kind == "script":

            target = (
                SCRIPT_DIR
                / product
            )

            label = "话术文件夹"

        elif kind == "wav":

            target = (
                OUTPUT_DIR
                / product
            )

            label = "WAV 文件夹"

        else:
            raise RuntimeError(
                "不支持的本地目录类型。"
            )

        if not target.exists():
            raise RuntimeError(
                f"{label}不存在，请先生成对应文件。"
            )

        if os.name == "nt":

            os.startfile(
                str(target)
            )

        elif (
            hasattr(os, "uname")
            and
            os.uname().sysname == "Darwin"
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
            "ok": True,
            "message":
                f"已打开「{product}」的{label}。",
            "path":
                str(target)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

# =========================================================
# 第8部分：自动播放 / 直播控制 / WAV生成 API
# =========================================================


# =========================================================
# 自动播放状态
# =========================================================

@app.get("/api/auto/status")
def api_auto_status():
    """
    自动播放状态。
    当前项目已经取消独立的 auto 播放状态，
    统一使用普通播放器状态。
    """
    return get_player_state()


@app.post("/api/auto/start")
def api_auto_start(
    req: AutoPlayRequest
):
    try:
        product = validate_product_name(
            req.product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        # AI话术生成过程中禁止启动
        ai_state = get_ai_state()

        if ai_state.get("status") == "generating":
            raise RuntimeError(
                "AI 话术正在生成，请等待完成后再启动播放。"
            )

        # WAV生成过程中禁止启动
        tts_state = get_tts_state()

        if tts_state.get("status") in (
            "loading",
            "generating"
        ):
            raise RuntimeError(
                "WAV 正在生成，请等待完成后再启动播放。"
            )

        # 统一使用普通播放器
        start_playback(
            product
        )

        return get_player_state()

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@app.post("/api/auto/stop")
def api_auto_stop():
    try:
        stop_playback()

        return get_player_state()

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 停止自动播放
# =========================================================

@app.post("/api/auto/stop")
def api_auto_stop():

    try:

        stop_playback()

        return get_auto_state()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 开始直播
# =========================================================

@app.post("/api/play/start")
def api_start(
    req: ProductRequest
):
    try:

        product = validate_product_name(
            req.product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        # AI话术生成过程中禁止开始直播
        ai_state = get_ai_state()

        if ai_state.get("status") == "generating":
            raise RuntimeError(
                "AI 话术正在生成，请等待完成后再直播。"
            )

        # WAV生成过程中禁止开始直播
        tts_state = get_tts_state()

        if tts_state.get("status") in (
            "loading",
            "generating"
        ):
            raise RuntimeError(
                "WAV 正在生成，请等待完成后再直播。"
            )

        # 启动新的普通直播播放线程
        start_playback(
            product
        )

        return get_player_state()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 请求暂停直播
# =========================================================

@app.post("/api/play/pause")
def api_pause():

    try:

        # 这里调用新的暂停函数
        # 不立即打断正在播放的 WAV
        return pause_playback()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 恢复直播
# =========================================================

@app.post("/api/play/resume")
def api_resume():

    try:

        return resume_playback()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 停止直播
# =========================================================

@app.post("/api/play/stop")
def api_stop():

    try:

        stop_playback()

        return get_player_state()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# 生成当前商品全部 WAV
# =========================================================

@app.post("/api/tts/generate")
def api_tts_generate(
    req: ProductRequest
):
    try:

        product = validate_product_name(
            req.product
        )

        if product not in get_products():
            raise RuntimeError(
                f"商品「{product}」不存在。"
            )

        current = get_player_state()

        # 直播运行过程中不能生成 WAV
        if current.get("status") in (
            "starting",
            "playing",
            "pause_requested",
            "paused"
        ):
            raise RuntimeError(
                "直播正在运行，请先停止直播。"
            )

        # AI话术生成过程中不能生成 WAV
        current_ai = get_ai_state()

        if current_ai.get("status") == "generating":
            raise RuntimeError(
                "AI 话术正在生成，请等待完成后再生成 WAV。"
            )

        # 如果已经有TTS任务正在运行，也禁止重复启动
        current_tts = get_tts_state()

        if current_tts.get("status") in (
            "loading",
            "generating"
        ):
            raise RuntimeError(
                "WAV 正在生成，请勿重复点击。"
            )

        # 启动TTS生成线程
        start_tts_generation(
            product
        )

        return get_tts_state()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
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
     直播间 AI 实时问答
     ======================================================= -->

<div class="panel">

    <div class="panel-title">

        <h2>
            💬 直播 AI 实时问答
        </h2>

        <span class="small">
            AI回答不打断当前语音
        </span>

    </div>


    <div class="ai-message">

        <div>
            状态：
            <b id="liveAiStatus">
                等待观众提问
            </b>
        </div>

        <div style="margin-top:6px;">
            等待播放：
            <b id="liveAiQueue">
                0
            </b>
        </div>

    </div>


    <div
        style="
            margin-top:8px;
            padding:10px;
            border-radius:8px;
            background:#0d121b;
            line-height:1.6;
            font-size:13px;
        "
    >

        <div>
            <span style="color:#7f8ba0;">
                当前问题：
            </span>

            <span id="liveAiQuestion">
                暂无
            </span>
        </div>


        <div style="margin-top:8px;">

            <span style="color:#7f8ba0;">
                AI回答：
            </span>

            <span id="liveAiAnswer">
                暂无
            </span>

        </div>


        <div
            id="liveAiMessage"
            style="
                margin-top:8px;
                color:#8c99ad;
            "
        >
            等待观众提问
        </div>

    </div>


    <button
        class="tool-btn btn-purple"
        onclick="testLiveAi()"
        style="
            width:100%;
            margin-top:9px;
        "
    >
        🧪 测试直播 AI
    </button>


    <div
        class="ai-message"
        style="margin-top:8px;"
    >
        AI回答在后台生成。
        当前普通 WAV 播放期间不会被打断。
        当前 WAV 播放完成后，优先播放已经准备好的 AI回答。
        AI回答结束后继续普通话术。
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
   第9部分：前端 JavaScript
   商品 / 话术 / TTS / 自动播放 / 直播 / AI问答
   ========================================================= */


/* =========================================================
   全局状态
   ========================================================= */

let currentState = {
    status: "stopped",
    product: "",
    current_index: 0,
    total: 0,
    current_title: "",
    current_text: "",
    current_version: "",
    current_audio: "",
    current_audio_duration: 0,
    round_no: 0,
    progress_percent: 0,
    live_started_at: null,
    live_stopped_at: null,
    alert_level: "ok",
    alert_message: "",
    message: "等待开始直播"
};


let currentTtsState = {
    status: "idle",
    product: "",
    current: 0,
    total: 0,
    progress: 0,
    message: ""
};


let currentAiState = {
    status: "idle",
    product: "",
    style: "",
    current: 0,
    total: 0,
    progress: 0,
    message: "",
    error: ""
};


let currentAutoState = {
    status: "stopped",
    product: "",
    message: ""
};


let currentLiveAiState = {
    status: "idle",
    queue_size: 0,
    generating: false,
    current_question: "",
    current_answer: "",
    last_question: "",
    last_answer: "",
    last_status: "",
    message: "等待观众提问",
    error: ""
};


/* =========================================================
   通用 API 请求
   ========================================================= */

async function api(
    url,
    options = {}
) {

    const response =
        await fetch(
            url,
            {
                ...options,
                headers: {
                    "Content-Type":
                        "application/json",
                    ...(options.headers || {})
                }
            }
        );


    let data = null;

    try {

        data =
            await response.json();

    } catch (e) {

        data = null;

    }


    if (!response.ok) {

        let message =
            "请求失败";

        if (
            data &&
            data.detail
        ) {

            message =
                data.detail;

        } else if (
            data &&
            data.message
        ) {

            message =
                data.message;

        } else {

            message =
                response.status
                + " "
                + response.statusText;
        }

        throw new Error(
            message
        );
    }


    return data;
}


/* =========================================================
   DOM辅助
   ========================================================= */

function el(id) {

    return document.getElementById(
        id
    );

}


function setText(
    id,
    value
) {

    const node =
        el(id);

    if (node) {

        node.textContent =
            value == null
            ? ""
            : String(value);

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
            el("product");

        if (!select) {

            return;

        }


        const products =
            Array.isArray(
                data.products
            )
                ? data.products
                : [];


        const oldProduct =
            select.value;


        select.innerHTML = "";


        if (
            products.length === 0
        ) {

            const option =
                document.createElement(
                    "option"
                );

            option.value = "";

            option.textContent =
                "暂无商品";

            select.appendChild(
                option
            );

            return;

        }


        products.forEach(
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


        let target =
            preferredProduct
            ||
            oldProduct
            ||
            currentState.product;


        if (
            !products.includes(
                target
            )
        ) {

            target =
                products[0];

        }


        select.value =
            target;


        if (target) {

            await loadScript();

            await loadScriptVersions();

            await loadProductInfo();

            await refreshTtsState();

        }

    } catch (e) {

        console.error(
            "加载商品列表失败：",
            e
        );

    }

}


/* =========================================================
   当前商品
   ========================================================= */

function getSelectedProduct() {

    const select =
        el("product");

    if (!select) {

        return "";

    }

    return (
        select.value
        || ""
    ).trim();

}


/* =========================================================
   添加商品
   ========================================================= */

function addProduct() {

    const editor =
        el("productEditor");

    if (editor) {

        editor.style.display =
            "";

    }


    const nameInput =
        el("newProductName");

    if (nameInput) {

        nameInput.focus();

    }

}


/* =========================================================
   取消添加商品
   ========================================================= */

function cancelAddProduct() {

    const editor =
        el("productEditor");

    if (editor) {

        editor.style.display =
            "none";

    }

}


/* =========================================================
   确认添加商品
   ========================================================= */

async function confirmAddProduct() {

    const productInput =
        el("newProductName");

    if (!productInput) {

        alert(
            "找不到商品名称输入框。"
        );

        return;

    }


    const product =
        productInput.value.trim();


    if (!product) {

        alert(
            "请输入商品名称。"
        );

        productInput.focus();

        return;

    }


    const scriptInput =
        el("newProductScript");


    const payload = {

        product:
            product,

        script:
            scriptInput
                ? scriptInput.value
                : "",

        price:
            el("newProductPrice")
                ? el("newProductPrice").value
                : "",

        specification:
            el("newProductSpecification")
                ? el("newProductSpecification").value
                : "",

        selling_points:
            el("newProductSellingPoints")
                ? el("newProductSellingPoints").value
                : "",

        target_people:
            el("newProductTarget")
                ? el("newProductTarget").value
                : "",

        usage_scene:
            el("newProductScene")
                ? el("newProductScene").value
                : "",

        notes:
            el("newProductNotes")
                ? el("newProductNotes").value
                : "",

        extra:
            el("newProductExtra")
                ? el("newProductExtra").value
                : ""

    };


    try {

        const data =
            await api(
                "/api/product/create",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            payload
                        )
                }
            );


        alert(
            data.message
            ||
            "商品添加成功。"
        );


        cancelAddProduct();


        await loadProducts(
            data.product
        );


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   加载商品资料
   ========================================================= */

async function loadProductInfo() {

    const product =
        getSelectedProduct();

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
                +
                "/info"
            );


        const fields = {

            price:
                "infoPrice",

            specification:
                "infoSpecification",

            selling_points:
                "infoSellingPoints",

            target_people:
                "infoTarget",

            usage_scene:
                "infoScene",

            notes:
                "infoNotes",

            extra:
                "infoExtra"

        };


        Object.entries(
            fields
        ).forEach(
            ([key, id]) => {

                const node =
                    el(id);

                if (node) {

                    node.value =
                        data[key]
                        ||
                        "";

                }

            }
        );


    } catch (e) {

        console.error(
            "加载商品资料失败：",
            e
        );

    }

}


/* =========================================================
   保存商品资料
   ========================================================= */

async function saveProductInfo() {

    const product =
        getSelectedProduct();

    if (!product) {

        alert(
            "请先选择商品。"
        );

        return;

    }


    const payload = {

        product:
            product,

        price:
            el("infoPrice")
                ? el("infoPrice").value
                : "",

        specification:
            el("infoSpecification")
                ? el("infoSpecification").value
                : "",

        selling_points:
            el("infoSellingPoints")
                ? el("infoSellingPoints").value
                : "",

        target_people:
            el("infoTarget")
                ? el("infoTarget").value
                : "",

        usage_scene:
            el("infoScene")
                ? el("infoScene").value
                : "",

        notes:
            el("infoNotes")
                ? el("infoNotes").value
                : "",

        extra:
            el("infoExtra")
                ? el("infoExtra").value
                : ""

    };


    try {

        const data =
            await api(
                "/api/product/info",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            payload
                        )
                }
            );


        alert(
            data.message
            ||
            "商品资料已保存。"
        );


        await loadScript();

        await loadProductInfo();

        await refreshTtsState();


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
        getSelectedProduct();

    if (!product) {

        alert(
            "请先选择商品。"
        );

        return;

    }


    if (
        !confirm(
            "确定删除商品「"
            +
            product
            +
            "」吗？\n\n"
            +
            "该商品的话术和 WAV 文件也会一起删除。"
        )
    ) {

        return;

    }


    try {

        const data =
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


        alert(
            data.message
            ||
            "商品删除成功。"
        );


        await loadProducts();


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   打开本地话术 / WAV
   ========================================================= */

async function openLocalFolder(
    kind
) {

    const product =
        getSelectedProduct();

    if (!product) {

        alert(
            "请先选择商品。"
        );

        return;

    }


    try {

        const data =
            await api(
                "/api/local/open",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            {
                                product:
                                    product,
                                kind:
                                    kind
                            }
                        )
                }
            );


        if (
            data.message
        ) {

            console.log(
                data.message
            );

        }


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   加载当前商品话术
   ========================================================= */

async function loadScript() {

    const product =
        getSelectedProduct();

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


        const scriptBox =
            el("scriptContent");


        if (scriptBox) {

            scriptBox.value =
                data.script
                ||
                "";

        }


        const scriptText =
            el("scriptText");

        if (scriptText) {

            scriptText.textContent =
                data.script
                ||
                "";

        }


    } catch (e) {

        console.error(
            "加载话术失败：",
            e
        );

    }

}


/* =========================================================
   加载话术版本
   ========================================================= */

async function loadScriptVersions() {

    const product =
        getSelectedProduct();

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
            el("scriptVersion");

        if (!select) {

            return;

        }


        const versions =
            Array.isArray(
                data.versions
            )
                ? data.versions
                : [];


        const oldValue =
            select.value;


        select.innerHTML = "";


        versions.forEach(
            item => {

                let value = "";
                let label = "";


                if (
                    typeof item ===
                    "string"
                ) {

                    value =
                        item;

                    label =
                        item;

                } else {

                    value =
                        item.style
                        ||
                        item.version
                        ||
                        "";

                    label =
                        item.name
                        ||
                        item.style
                        ||
                        item.version
                        ||
                        "";

                }


                if (!value) {

                    return;

                }


                const option =
                    document.createElement(
                        "option"
                    );

                option.value =
                    value;

                option.textContent =
                    label;

                select.appendChild(
                    option
                );

            }
        );


        if (
            oldValue &&
            Array.from(
                select.options
            ).some(
                option =>
                    option.value ===
                    oldValue
            )
        ) {

            select.value =
                oldValue;

        }


    } catch (e) {

        console.error(
            "加载话术版本失败：",
            e
        );

    }

}


/* =========================================================
   获取当前选择的话术风格
   ========================================================= */

function getSelectedStyle() {

    const select =
        el("scriptVersion");

    if (!select) {

        return "";

    }

    return (
        select.value
        || ""
    ).trim();

}


/* =========================================================
   AI重新生成当前话术
   ========================================================= */

async function regenerateScript() {

    const product =
        getSelectedProduct();

    if (!product) {

        alert(
            "请先选择商品。"
        );

        return;

    }


    const style =
        getSelectedStyle();


    try {

        const data =
            await api(
                "/api/script/regenerate",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            {
                                product:
                                    product,
                                style:
                                    style
                            }
                        )
                }
            );


        renderAiState(
            data
        );


        alert(
            "AI话术生成任务已启动。"
        );


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   AI生成全部话术
   ========================================================= */

async function regenerateAllScripts() {

    const product =
        getSelectedProduct();

    if (!product) {

        alert(
            "请先选择商品。"
        );

        return;

    }


    if (
        !confirm(
            "确定为商品「"
            +
            product
            +
            "」生成全部话术风格吗？"
        )
    ) {

        return;

    }


    try {

        const data =
            await api(
                "/api/script/regenerate-all",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            {
                                product:
                                    product
                            }
                        )
                }
            );


        renderAiState(
            data
        );


        alert(
            "全部话术生成任务已启动。"
        );


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   AI话术状态刷新
   ========================================================= */

async function refreshAiState() {

    try {

        const data =
            await api(
                "/api/script/status"
            );


        currentAiState =
            data
            ||
            currentAiState;


        renderAiState(
            currentAiState
        );


    } catch (e) {

        console.error(
            "刷新AI话术状态失败：",
            e
        );

    }

}


/* =========================================================
   显示AI话术状态
   ========================================================= */

function renderAiState(
    state
) {

    if (!state) {

        return;

    }


    const status =
        state.status
        ||
        "idle";


    let text =
        state.message
        ||
        "等待操作";


    if (
        status ===
        "generating"
    ) {

        text =
            state.message
            ||
            "AI正在生成话术……";

    }


    setText(
        "aiStatus",
        text
    );


    setText(
        "aiProgress",
        state.progress != null
            ? state.progress + "%"
            : ""
    );


    const progress =
        el("aiProgressBar");


    if (progress) {

        const value =
            Number(
                state.progress
                ||
                0
            );

        progress.style.width =
            Math.max(
                0,
                Math.min(
                    100,
                    value
                )
            )
            + "%";

    }

}


/* =========================================================
   选择话术版本
   ========================================================= */

async function selectScriptVersion() {

    const product =
        getSelectedProduct();

    const style =
        getSelectedStyle();


    if (
        !product ||
        !style
    ) {

        return;

    }


    try {

        const data =
            await api(
                "/api/script/select",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            {
                                product:
                                    product,
                                style:
                                    style
                            }
                        )
                }
            );


        alert(
            data.message
            ||
            "话术版本已切换。"
        );


        await loadScript();

        await refreshTtsState();


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   TTS状态刷新
   ========================================================= */

async function refreshTtsState() {

    try {

        const data =
            await api(
                "/api/tts/status"
            );


        currentTtsState =
            data
            ||
            currentTtsState;


        renderTtsState(
            currentTtsState
        );


    } catch (e) {

        console.error(
            "刷新TTS状态失败：",
            e
        );

    }

}


/* =========================================================
   显示TTS状态
   ========================================================= */

function renderTtsState(
    state
) {

    if (!state) {

        return;

    }


    let message =
        state.message
        ||
        "等待生成 WAV";


    if (
        state.status ===
        "generating"
    ) {

        message =
            state.message
            ||
            "正在生成 WAV……";

    } else if (
        state.status ===
        "loading"
    ) {

        message =
            state.message
            ||
            "正在加载语音模型……";

    } else if (
        state.status ===
        "done"
    ) {

        message =
            state.message
            ||
            "WAV生成完成";

    } else if (
        state.status ===
        "error"
    ) {

        message =
            state.message
            ||
            "WAV生成失败";

    }


    setText(
        "ttsStatus",
        message
    );


    const progressValue =
        Number(
            state.progress
            ||
            0
        );


    setText(
        "ttsProgress",
        progressValue
        + "%"
    );


    const progressBar =
        el("ttsProgressBar");


    if (progressBar) {

        progressBar.style.width =
            Math.max(
                0,
                Math.min(
                    100,
                    progressValue
                )
            )
            + "%";

    }

}


/* =========================================================
   生成全部WAV
   ========================================================= */

async function generateAllWav() {

    const product =
        getSelectedProduct();

    if (!product) {

        alert(
            "请先选择商品。"
        );

        return;

    }


    try {

        const data =
            await api(
                "/api/tts/generate",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            {
                                product:
                                    product
                            }
                        )
                }
            );


        currentTtsState =
            data;


        renderTtsState(
            data
        );


        alert(
            "WAV生成任务已启动。"
        );


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   自动播放状态
   ========================================================= */

async function refreshAutoState() {

    try {

        const data =
            await api(
                "/api/auto/status"
            );


        currentAutoState =
            data
            ||
            currentAutoState;


        renderAutoState(
            currentAutoState
        );


    } catch (e) {

        console.error(
            "刷新自动播放状态失败：",
            e
        );

    }

}


/* =========================================================
   显示自动播放状态
   ========================================================= */

function renderAutoState(
    state
) {

    if (!state) {

        return;

    }


    let text =
        state.message
        ||
        "自动播放未启动";


    if (
        state.status ===
        "playing"
    ) {

        text =
            state.message
            ||
            "自动播放中";

    } else if (
        state.status ===
        "stopped"
    ) {

        text =
            state.message
            ||
            "自动播放已停止";

    }


    setText(
        "autoStatus",
        text
    );

}


/* =========================================================
   启动自动播放
   ========================================================= */

async function startAutoPlay() {

    const product =
        getSelectedProduct();

    if (!product) {

        alert(
            "请先选择商品。"
        );

        return;

    }


    try {

        const data =
            await api(
                "/api/auto/start",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            {
                                product:
                                    product
                            }
                        )
                }
            );


        currentAutoState =
            data;


        renderAutoState(
            data
        );


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   停止自动播放
   ========================================================= */

async function stopAutoPlay() {

    try {

        const data =
            await api(
                "/api/auto/stop",
                {
                    method:
                        "POST"
                }
            );


        currentAutoState =
            data;


        renderAutoState(
            data
        );


        await refreshState();

    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   开始直播
   ========================================================= */

async function startLive() {

    const product =
        getSelectedProduct();

    if (!product) {

        alert(
            "请先选择商品。"
        );

        return;

    }


    try {

        const data =
            await api(
                "/api/play/start",
                {
                    method: "POST",
                    body:
                        JSON.stringify(
                            {
                                product:
                                    product
                            }
                        )
                }
            );


        currentState =
            data;


        render(
            data
        );


        await refreshLiveAiState();


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   暂停直播
   ========================================================= */

async function pauseLive() {

    try {

        const data =
            await api(
                "/api/play/pause",
                {
                    method:
                        "POST"
                }
            );


        currentState =
            data;


        render(
            data
        );


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   恢复直播
   ========================================================= */

async function resumeLive() {

    try {

        const data =
            await api(
                "/api/play/resume",
                {
                    method:
                        "POST"
                }
            );


        currentState =
            data;


        render(
            data
        );


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   停止直播
   ========================================================= */

async function stopLive() {

    try {

        const data =
            await api(
                "/api/play/stop",
                {
                    method:
                        "POST"
                }
            );


        currentState =
            data;


        render(
            data
        );


        await refreshLiveAiState();


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


        currentState =
            state;


        render(
            state
        );


    } catch (e) {

        console.error(
            "刷新直播状态失败：",
            e
        );

    }

}


/* =========================================================
   渲染直播状态
   ========================================================= */

function render(
    state
) {

    if (!state) {

        return;

    }


    const status =
        state.status
        ||
        "stopped";


    let statusText =
        "已停止";


    if (
        status ===
        "starting"
    ) {

        statusText =
            "正在启动";

    } else if (
        status ===
        "playing"
    ) {

        statusText =
            "直播播放中";

    } else if (
        status ===
        "pause_requested"
    ) {

        statusText =
            "即将暂停";

    } else if (
        status ===
        "paused"
    ) {

        statusText =
            "已暂停";

    } else if (
        status ===
        "error"
    ) {

        statusText =
            "播放异常";

    }


    setText(
        "liveStatus",
        statusText
    );


    setText(
        "currentProduct",
        state.product
        ||
        "-"
    );


    setText(
        "currentTitle",
        state.current_title
        ||
        "-"
    );


    setText(
        "currentVersion",
        state.current_version
        ||
        "-"
    );


    setText(
        "currentIndex",
        state.current_index
        || 0
    );


    setText(
        "totalCount",
        state.total
        || 0
    );


    setText(
        "roundNo",
        state.round_no
        || 0
    );


    setText(
        "currentAudio",
        state.current_audio
        ||
        "-"
    );


    setText(
        "currentAudioDuration",
        state.current_audio_duration
        ?
        state.current_audio_duration
        + " 秒"
        :
        "-"
    );


    setText(
        "liveMessage",
        state.alert_message
        ||
        state.message
        ||
        ""
    );


    const progress =
        Number(
            state.progress_percent
            ||
            0
        );


    setText(
        "liveProgress",
        progress
        + "%"
    );


    const progressBar =
        el("liveProgressBar");


    if (progressBar) {

        progressBar.style.width =
            Math.max(
                0,
                Math.min(
                    100,
                    progress
                )
            )
            + "%";

    }


    updateLiveButtons(
        status
    );

}


/* =========================================================
   更新直播按钮状态
   ========================================================= */

function updateLiveButtons(
    status
) {

    const startBtn =
        el("startLiveBtn");

    const pauseBtn =
        el("pauseLiveBtn");

    const resumeBtn =
        el("resumeLiveBtn");

    const stopBtn =
        el("stopLiveBtn");


    if (startBtn) {

        startBtn.disabled =
            status === "starting"
            ||
            status === "playing"
            ||
            status === "pause_requested"
            ||
            status === "paused";

    }


    if (pauseBtn) {

        pauseBtn.disabled =
            !(
                status ===
                "playing"
            );

    }


    if (resumeBtn) {

        resumeBtn.disabled =
            !(
                status ===
                "paused"
            );

    }


    if (stopBtn) {

        stopBtn.disabled =
            status ===
            "stopped";

    }

}


/* =========================================================
   直播运行时间
   ========================================================= */

function updateLiveDuration() {

    const node =
        el("liveDuration");

    if (!node) {

        return;

    }


    if (
        !currentState.live_started_at
    ) {

        node.textContent =
            "00:00:00";

        return;

    }


    if (
        currentState.status ===
        "stopped"
    ) {

        node.textContent =
            "00:00:00";

        return;

    }


    const start =
        new Date(
            currentState.live_started_at
        );


    if (
        Number.isNaN(
            start.getTime()
        )
    ) {

        return;

    }


    const now =
        new Date();


    let seconds =
        Math.floor(
            (
                now.getTime()
                -
                start.getTime()
            )
            /
            1000
        );


    if (
        seconds < 0
    ) {

        seconds = 0;

    }


    const h =
        Math.floor(
            seconds / 3600
        );


    const m =
        Math.floor(
            (
                seconds % 3600
            ) / 60
        );


    const s =
        seconds % 60;


    node.textContent =
        String(h).padStart(
            2,
            "0"
        )
        +
        ":"
        +
        String(m).padStart(
            2,
            "0"
        )
        +
        ":"
        +
        String(s).padStart(
            2,
            "0"
        );

}


/* =========================================================
   系统时间
   ========================================================= */

function updateClock() {

    const node =
        el("clock");

    if (!node) {

        return;

    }


    const now =
        new Date();


    const y =
        now.getFullYear();


    const mo =
        String(
            now.getMonth() + 1
        ).padStart(
            2,
            "0"
        );


    const d =
        String(
            now.getDate()
        ).padStart(
            2,
            "0"
        );


    const h =
        String(
            now.getHours()
        ).padStart(
            2,
            "0"
        );


    const m =
        String(
            now.getMinutes()
        ).padStart(
            2,
            "0"
        );


    const s =
        String(
            now.getSeconds()
        ).padStart(
            2,
            "0"
        );


    node.textContent =
        y
        +
        "-"
        +
        mo
        +
        "-"
        +
        d
        +
        " "
        +
        h
        +
        ":"
        +
        m
        +
        ":"
        +
        s;

}


/* =========================================================
   直播 AI 问答状态
   ========================================================= */

async function refreshLiveAiState() {

    try {

        const data =
            await api(
                "/api/live/ai/status"
            );


        currentLiveAiState =
            data
            ||
            currentLiveAiState;


        renderLiveAiState(
            currentLiveAiState
        );


    } catch (e) {

        console.error(
            "刷新直播AI状态失败：",
            e
        );

    }

}


/* =========================================================
   渲染直播 AI 状态
   ========================================================= */

function renderLiveAiState(
    state
) {

    if (!state) {

        return;

    }


    const status =
        state.status
        ||
        "idle";


    let statusText =
        "等待观众提问";


    if (
        status ===
        "generating"
    ) {

        statusText =
            "🧠 AI正在生成回答";

    } else if (
        status ===
        "queued"
    ) {

        statusText =
            "⏳ AI回答等待播放";

    } else if (
        status ===
        "playing"
    ) {

        statusText =
            "🔊 正在播放AI回答";

    } else if (
        status ===
        "error"
    ) {

        statusText =
            "❌ AI回答异常";

    }


    setText(
        "liveAiStatus",
        statusText
    );


    setText(
        "liveAiQueue",
        state.queue_size
        || 0
    );


    setText(
        "liveAiQuestion",
        state.current_question
        ||
        state.last_question
        ||
        "暂无"
    );


    setText(
        "liveAiAnswer",
        state.current_answer
        ||
        state.last_answer
        ||
        "暂无"
    );


    setText(
        "liveAiMessage",
        state.error
        ?
        "❌ " + state.error
        :
        (
            state.message
            ||
            "等待观众提问"
        )
    );

}


/* =========================================================
   测试直播 AI
   ========================================================= */

async function testLiveAi() {

    try {

        const data =
            await api(
                "/api/live/ai/test",
                {
                    method:
                        "POST"
                }
            );


        currentLiveAiState =
            data;


        renderLiveAiState(
            data
        );


    } catch (e) {

        alert(
            e.message
        );

    }

}


/* =========================================================
   手动选择商品
   ========================================================= */

const productSelect =
    el("product");


if (productSelect) {

    productSelect.addEventListener(
        "change",
        async function () {

            const product =
                productSelect.value;


            if (!product) {

                return;

            }


            try {

                /*
                   这里只加载当前选择的商品。

                   不自动切换到其他商品。
                   不自动开始播放。
                   不自动生成下一轮话术。
                */

                await loadScript();

                await loadScriptVersions();

                await loadProductInfo();

                await refreshTtsState();

            } catch (e) {

                console.error(
                    "切换商品失败：",
                    e
                );

            }

        }
    );

}


/* =========================================================
   话术版本选择
   ========================================================= */

const scriptVersionSelect =
    el("scriptVersion");


if (scriptVersionSelect) {

    scriptVersionSelect.addEventListener(
        "change",
        selectScriptVersion
    );

}


/* =========================================================
   初始化
   ========================================================= */

async function init() {

    try {

        await loadProducts();

        await refreshState();

        await refreshTtsState();

        await refreshAiState();

        await refreshAutoState();

        await refreshLiveAiState();

        updateClock();

        updateLiveDuration();

    } catch (e) {

        console.error(
            "控制台初始化失败：",
            e
        );

    }

}


/* =========================================================
   启动初始化
   ========================================================= */

init();


/* =========================================================
   定时刷新
   ========================================================= */


/*
   普通直播播放器状态
*/
setInterval(
    refreshState,
    500
);


/*
   TTS WAV生成状态
*/
setInterval(
    refreshTtsState,
    500
);


/*
   AI话术生成状态
*/
setInterval(
    refreshAiState,
    500
);


/*
   自动播放状态
*/
setInterval(
    refreshAutoState,
    500
);


/*
   直播观众AI问答状态
*/
setInterval(
    refreshLiveAiState,
    500
);


/*
   直播运行时间
*/
setInterval(
    updateLiveDuration,
    1000
);


/*
   系统时间
*/
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

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "AI直播控制台"
    )

    print(
        "访问地址：http://127.0.0.1:8000"
    )

    print(
        "AI直播问答接口：http://127.0.0.1:8001/comment"
    )

    print(
        "按 Ctrl+C 退出"
    )

    print(
        "=" * 60
    )


    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )

