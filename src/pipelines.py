from datetime import datetime
import pandas as pd
from .get_reddit_data import get_post_data
from .sentiment_analysis import get_sentiment
from .text_processor import clean_text
import json
import os

# 读取 Reddit Client 配置 (假设你原来的代码是这么初始化的)
# 如果你原来的 pipelines.py 里有特殊的 reddit 初始化逻辑，请保留
# 这里为了简洁，直接实例化 praw，你需要确保这里的鉴权逻辑和你项目里的一致
import praw
def get_reddit_instance():
    # 尝试从环境变量读取 (GitHub Action 模式)
    if os.environ.get("REDDIT_CLIENT_ID"):
        return praw.Reddit(
            client_id=os.environ.get("REDDIT_CLIENT_ID"),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
            user_agent=os.environ.get("REDDIT_USER_AGENT")
        )
    # 或者从本地文件读取 (本地开发模式)
    # ... (保留你原有的逻辑)
    return None

reddit = get_reddit_instance()

def convert_utc(utc_time):
    return datetime.utcfromtimestamp(utc_time)

def top_posts_subreddit_pipeline(
    subreddit_name, post_limit, comment_limmit, posts_to_get="Hot"
):
    # 透传 posts_to_get 参数
    post_data = get_post_data(
        subreddit_name=subreddit_name,
        post_limit=post_limit,
        comment_limmit=comment_limmit,
        reddit=reddit, # 传入 reddit 实例
        posts_to_get=posts_to_get, # <--- 关键修改
    )
    
    df = pd.DataFrame(post_data)
    if df.empty: return df

    df["all_text"] = df["title"] + df["selftext"]
    df["clean_title"] = df["all_text"].apply(lambda x: clean_text(x))
    
    # 执行情绪分析
    df = get_sentiment(df, "clean_title")
    
    df["timestamp"] = df["created_utc"].apply(convert_utc)
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day

    return df
