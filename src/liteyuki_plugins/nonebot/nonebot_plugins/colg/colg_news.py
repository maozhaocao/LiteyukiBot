import os
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup
from nonebot import require
from nonebot_plugin_apscheduler import scheduler

from liteyuki.config import load_from_json
from src.utils.message.message import broadcast_to_superusers

require("nonebot_plugin_apscheduler")

url_list = [
    "https://bbs.colg.cn/home.php?mod=space&uid=4120473",
    "https://bbs.colg.cn/home.php?mod=space&uid=80727",
]

author_list = ["白猫之惩戒", "魔法少女QB"]
keywords = ["韩服", "爆料", "国服", "前瞻", "韩测"]
limit = 6
pre_head = []
# proxies = load_from_json(os.path.realpath("./proxies2.json"))
proxies = {}

def fetch_content(url):
    try:
        response = requests.get(url, proxies=proxies, timeout=5, verify=False)
        response.raise_for_status()  # 将触发HTTPError，如果状态不是200
    except requests.RequestException as e:
        return "", "", str(e)

    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find("div", id="thread_content")  # 找到id为thread_content的div
    if content_div is None:
        return "", "", "未找到id为thread_content的元素"

    ul_element = content_div.find("ul")  # 在div内查找ul元素
    if ul_element is None:
        return "", "", "未找到ul元素"

    context = []
    head = ""

    for i, li in enumerate(ul_element.find_all("li")):  # 在ul元素内查找所有li元素
        if i < limit:
            context.append(li.get_text() + "\r\n")
            for link in li.find_all("a"):
                context.append("https://bbs.colg.cn" + link.get("href") + "\r\n")
            if i == 0:
                head = li.get_text()
    return "\r\n".join(context), head, None


def colg_news():
    context = []
    for url in url_list:
        tmp_context, _, err = fetch_content(url)
        if err:
            return "", err
        context.append(tmp_context)
    return "\r\n".join(context), None


def get_colg_change():
    new_list = []
    if not pre_head:
        for url in url_list:
            _, tmp_head, err = fetch_content(url)
            if err:
                return [], err
            pre_head.append(tmp_head)
        return [], None

    for order, url in enumerate(url_list):
        context, head, err = fetch_content(url)
        if err:
            return [], err
        if head != pre_head[order]:
            for keyword in keywords:
                if keyword in head and SequenceMatcher(None, head, pre_head[order]).ratio() < 0.8:
                    context = "colg资讯已更新:\r\n" + context + author_list[order]
                    new_list.append(context)
                    break
            pre_head[order] = head

    return new_list, None


@scheduler.scheduled_job("cron", minute=5)
# @scheduler.scheduled_job("cron", second=5)
async def every_day_update():
    new_list, _ = get_colg_change()
    if new_list:
        print(new_list)
        await broadcast_to_superusers("".join(new_list))
