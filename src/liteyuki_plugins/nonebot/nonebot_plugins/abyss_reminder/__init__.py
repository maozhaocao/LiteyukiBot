from nonebot.plugin import PluginMetadata

from .main import *

__author__ = "maozhaocao"
__plugin_meta__ = PluginMetadata(
    name="深渊小助手",
    description="",
    usage=(
        ""
    ),
    type="application",
    homepage="https://github.com/snowykami/LiteyukiBot",
    extra={
        "liteyuki": True,
        "toggleable": False,
        "default_enable": True,
    }
)

