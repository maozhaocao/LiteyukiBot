import nonebot
from fastapi import APIRouter
from fastapi import FastAPI
from nonebot import get_app, Bot

from src.utils.base.config import get_config

app: FastAPI = get_app()

abyss_router = APIRouter(prefix="/abyss")

abyss_token = get_config("abyss_api_token", None)


@abyss_router.get("/send_abyss_info")
async def send_abyss_info(user_id, msg, token):
    if abyss_token and abyss_token != token:
        return "unauthorized"
    bots: dict[str, Bot] = nonebot.get_bots()
    for bot in bots.values():
        await bot.send_private_msg(user_id=user_id, message=msg)
    return "success"


app.include_router(abyss_router)

