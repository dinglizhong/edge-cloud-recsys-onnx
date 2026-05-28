import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Load environment variables
load_dotenv()

# Initialize Kimi (Moonshot) client
client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)

# Define the expected output structure using Pydantic
class AppFeatures(BaseModel):
    category: str = Field(description="The primary category of the App (e.g., RPG Game, Tool, Social)")
    target_audience: List[str] = Field(description="List of target audience characteristics (e.g., ['Students', 'Gamer', 'Young Adults'])")
    core_features: List[str] = Field(description="Top 3 core features or selling points of the App")
    scenario: List[str] = Field(description="Typical usage scenarios (e.g., ['Commuting', 'Before sleep', 'Weekend'])")
    intent_tags: List[str] = Field(description="List of user intent tags for search retrieval (e.g., ['Kill time', 'Relax', 'Multiplayer'])")

def extract_features_with_llm(app_name: str, app_description: str) -> AppFeatures:
    """
    Use LLM to extract structured features from raw App descriptions.
    """
    prompt = f"""
    You are an expert AI data annotator working for OPPO App Store.
    Your task is to analyze the following App name and description, and extract structured features for our recommendation system.
    
    App Name: {app_name}
    App Description: {app_description}
    
    Please provide the extracted features strictly following the required JSON schema.
    Do NOT wrap the output in any extra keys like "features" or "app_name".
    Return a flat JSON object with EXACTLY these keys: "category", "target_audience", "core_features", "scenario", "intent_tags".
    IMPORTANT: "target_audience", "core_features", "scenario", and "intent_tags" MUST be JSON arrays (lists) of strings, even if there is only one item.
    Ensure the tags are concise and suitable for downstream machine learning models.
    """
    
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    
    # Parse the JSON response
    result_json = response.choices[0].message.content
    try:
        # Validate and convert using Pydantic
        features = AppFeatures.model_validate_json(result_json)
        return features
    except Exception as e:
        print(f"Error parsing LLM output for {app_name}: {e}")
        print(f"Raw output: {result_json}")
        return None

def process_mock_data():
    """
    Process a list of mock App data and save the enriched features.
    """
    # Mock data simulating OPPO App Store raw data
    mock_apps = [
        {
            "app_id": "A001",
            "app_name": "欢乐消消消",
            "description": "超好玩的休闲消除游戏！精美的画面，数百个关卡等你来挑战。随时随地，无需网络也能玩，是你打发无聊时间的最佳选择。快来和好友一起比拼分数吧！"
        },
        {
            "app_id": "A002",
            "app_name": "极速清理大师",
            "description": "手机卡顿？内存不足？极速清理大师一键帮你解决！深度清理微信/QQ缓存，释放手机空间，让你的手机快如闪电。同时提供电池优化、CPU降温功能。"
        },
        {
            "app_id": "A003",
            "app_name": "职场进阶课",
            "description": "专为职场新人打造的在线学习平台。汇聚行业大咖，提供PPT制作、演讲技巧、职场沟通等实战课程。每天碎片时间学习15分钟，助你快速升职加薪。"
        },
        {
            "app_id": "A004",
            "app_name": "王者荣耀",
            "description": "腾讯正版战术竞技手游，5V5英雄公平对战。海量英雄随心选择，精妙配合默契作战！10秒实时跨区匹配，与好友组队登顶最强王者！"
        },
        {
            "app_id": "A005",
            "app_name": "抖音",
            "description": "记录美好生活！超火的短视频社交平台，在这里你可以看到各种搞笑、才艺、生活分享。智能推荐算法，让你越刷越懂你，还能自己开直播带货。"
        },
        {
            "app_id": "A006",
            "app_name": "美颜相机",
            "description": "超千万女性用户的选择！智能美颜，一键拍出水光肌。提供海量滤镜、贴纸、AR特效，支持视频美颜和后期精修，让你的每一张自拍都完美无瑕。"
        },
        {
            "app_id": "A007",
            "app_name": "原神",
            "description": "米哈游出品的开放世界冒险游戏。在提瓦特大陆，你可以踏遍七国，邂逅性格各异、能力独特的同伴，与他们一同对抗强敌，踏上寻回血亲之路。"
        },
        {
            "app_id": "A008",
            "app_name": "网易云音乐",
            "description": "超清音质、优质歌单、精准推荐的音乐APP。不仅能听歌，还有超火的音乐社区，看动人热评，分享听歌心情。支持一键导入外部歌单。"
        }
    ]
    
    print("Starting LLM feature extraction...")
    enriched_data = []
    
    for app in mock_apps:
        print(f"\nProcessing App: {app['app_name']}")
        features = extract_features_with_llm(app["app_name"], app["description"])
        
        if features:
            # Combine original data with extracted features
            app_data = app.copy()
            app_data.update(features.model_dump())
            enriched_data.append(app_data)
            print(f"Extracted features: {features.model_dump_json(indent=2)}")
            
    # Save to a DataFrame for downstream model training
    df = pd.DataFrame(enriched_data)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/app_features_enriched.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nFeature extraction complete. Data saved to {output_path}")
    return df

if __name__ == "__main__":
    process_mock_data()
