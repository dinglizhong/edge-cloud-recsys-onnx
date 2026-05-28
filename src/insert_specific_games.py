import pandas as pd
import os

def insert_specific_games():
    new_apps = [
        {
            "app_id": "A100",
            "app_name": "和平精英",
            "description": "腾讯光子工作室群自研打造的战术竞技手游。虚幻引擎4研发，次世代完美画质，极致视听感受；超大实景地图，打造指尖战场，全方面自由施展战术；百人同场竞技，真实弹道，完美的射击手感。",
            "category": "Gaming",
            "target_audience": "['hardcore players', 'FPS lovers', 'youth']",
            "core_features": "['tactical competition', '100-player battle royale', 'voice chat', 'realistic ballistics']",
            "scenario": "['party with friends', 'weekend gaming', 'e-sports']",
            "intent_tags": "['shooting', 'survival', 'teamwork', 'competitive']"
        },
        {
            "app_id": "A101",
            "app_name": "三角洲行动",
            "description": "一款设定在2035年的第一人称特战干员战术射击游戏。支持多平台免费游玩。拥有极其硬核的改枪系统和多元化的战术道具，带你体验极致的现代战争。",
            "category": "Gaming",
            "target_audience": "['military enthusiasts', 'hardcore gamers', 'FPS lovers']",
            "core_features": "['first-person shooter', 'weapon customization', 'tactical operators']",
            "scenario": "['immersive gaming', 'weekend relax', 'stress relief']",
            "intent_tags": "['action', 'military', 'hardcore', 'shooting']"
        },
        {
            "app_id": "A102",
            "app_name": "部落冲突",
            "description": "加入全球数百万玩家的行列，建立村庄、组建部落，参加史诗般的部落对战！在策略与养成并重的世界里，训练野蛮人等个性化军队，与其他玩家一决高下。",
            "category": "Strategy",
            "target_audience": "['strategy lovers', 'casual gamers', 'global players']",
            "core_features": "['base building', 'clan wars', 'resource management', 'tower defense']",
            "scenario": "['daily check-in', 'killing time', 'social interaction']",
            "intent_tags": "['strategy', 'simulation', 'social', 'casual']"
        },
        {
            "app_id": "A103",
            "app_name": "王国保卫战",
            "description": "备受赞誉的塔防神作！在森林、山脉和荒地中指挥你的军队，通过升级定制的防御塔和特殊技能，抵御兽人、巨魔、邪恶巫师的入侵。",
            "category": "Strategy",
            "target_audience": "['tower defense fans', 'strategy lovers', 'single players']",
            "core_features": "['tower defense', 'hero system', 'offline play', 'challenging levels']",
            "scenario": "['commuting', 'solo entertainment', 'focus time']",
            "intent_tags": "['tower defense', 'strategy', 'offline', 'classic']"
        },
        {
            "app_id": "A104",
            "app_name": "愤怒的小鸟",
            "description": "风靡全球的经典物理弹射游戏。为了报复偷走鸟蛋的贪婪绿猪，玩家需要控制不同种类的小鸟，利用物理法则摧毁绿猪的堡垒。休闲解压，百玩不厌。",
            "category": "Games",
            "target_audience": "['all ages', 'kids', 'casual gamers']",
            "core_features": "['physics-based puzzle', 'slingshot mechanics', 'funny animations', 'offline play']",
            "scenario": "['restroom time', 'waiting in line', 'short breaks']",
            "intent_tags": "['puzzle', 'casual', 'physics', 'relaxing']"
        },
        {
            "app_id": "A105",
            "app_name": "保卫萝卜",
            "description": "国民级萌系塔防单机游戏！画风Q萌可爱，老少皆宜。通过建造各种奇特的炮塔，保护可爱的小萝卜免受怪物的袭击。数百个特色关卡，带来轻松愉快的体验。",
            "category": "Games",
            "target_audience": "['female users', 'kids', 'casual gamers', 'all ages']",
            "core_features": "['cute graphics', 'casual tower defense', 'level progression', 'offline play']",
            "scenario": "['before sleep', 'killing time', 'family time']",
            "intent_tags": "['cute', 'tower defense', 'casual', 'relax']"
        }
    ]

    csv_path = "data/app_features_enriched.csv"
    df = pd.read_csv(csv_path)
    
    # 避免重复插入
    existing_names = set(df['app_name'].tolist())
    to_add = [app for app in new_apps if app['app_name'] not in existing_names]
    
    if to_add:
        df_new = pd.DataFrame(to_add)
        df_all = pd.concat([df_new, df], ignore_index=True) # 放在最前面容易被召回
        df_all.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Added {len(to_add)} specific games. Total rows: {len(df_all)}")
    else:
        print("Games already exist in the dataset.")

if __name__ == "__main__":
    insert_specific_games()
