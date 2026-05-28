import pandas as pd
import random
import os

def generate_mock_apps(start_id, count):
    categories = ["Games", "Utilities", "Education", "Video", "Social", "News", "Tool", "Travel", "Shopping", "Finance", "Music", "Books"]
    prefixes = ["全民", "天天", "极速", "智能", "超级", "神奇", "懒人", "掌上", "口袋", "欢乐", "阳光", "星辰", "闪电", "优享", "畅快"]
    suffixes = ["助手", "管家", "大作战", "宝典", "神器", "精英", "世界", "日记", "大全", "专家", "卫士", "大师", "联盟", "先锋"]
    mid_words = {
        "Games": "消消", "Utilities": "清理", "Education": "课堂", "Video": "影院", 
        "Social": "交友", "News": "头条", "Tool": "工具", "Travel": "出行", 
        "Shopping": "优选", "Finance": "财富", "Music": "音乐", "Books": "阅读"
    }
    
    # --- 细分特征池库 ---
    target_audiences = {
        "Games": ["casual gamers", "hardcore players", "students", "youth", "e-sports fans"],
        "Utilities": ["smartphone users", "elderly", "office workers", "android users"],
        "Education": ["students", "parents", "professionals", "kids", "lifelong learners"],
        "Video": ["movie lovers", "drama fans", "anime fans", "vloggers", "gen z"],
        "Social": ["singles", "youth", "strangers", "hobbyists", "professionals"],
        "News": ["commuters", "elderly", "investors", "general public", "sports fans"],
        "Tool": ["office workers", "creators", "students", "drivers", "designers"],
        "Travel": ["tourists", "business travelers", "backpackers", "families", "drivers"],
        "Shopping": ["budget shoppers", "fashion lovers", "housewives", "trendsetters", "sneakerheads"],
        "Finance": ["investors", "bank customers", "young adults", "business owners", "crypto traders"],
        "Music": ["music lovers", "audiophiles", "singers", "commuters", "k-pop fans"],
        "Books": ["novel readers", "comic fans", "audiobook listeners", "students", "literature lovers"]
    }
    
    core_features = {
        "Games": ["multiplayer mode", "3D graphics", "auto-battle", "social ranking", "guild system", "story mode"],
        "Utilities": ["junk clean", "virus scan", "battery save", "CPU cool", "app lock", "network test"],
        "Education": ["live classes", "Q&A forum", "mock exams", "AI tutor", "progress tracking", "video courses"],
        "Video": ["4K streaming", "danmaku", "offline download", "screen cast", "live chat", "short clips"],
        "Social": ["voice chat", "matching algorithm", "virtual avatars", "group chat", "moments", "location sharing"],
        "News": ["personalized feed", "breaking news alerts", "audio reading", "comment section", "offline reading"],
        "Tool": ["OCR scanning", "cloud sync", "widgets", "data export", "password protect", "multi-device"],
        "Travel": ["route planning", "ticket booking", "travel guides", "real-time transit", "AR navigation"],
        "Shopping": ["live commerce", "flash sales", "price tracking", "AR try-on", "secure payment", "buyer reviews"],
        "Finance": ["real-time quotes", "auto-invest", "expense tracking", "secure login", "market analysis"],
        "Music": ["lossless audio", "lyrics sync", "karaoke mode", "smart radio", "equalizer", "playlist import"],
        "Books": ["eye-care mode", "text-to-speech", "offline reading", "bookmark sync", "community reviews"]
    }
    
    scenarios = {
        "Games": ["killing time", "party with friends", "weekend relax", "commuting", "stress relief"],
        "Utilities": ["phone lagging", "low storage", "device overheating", "public wifi", "daily maintenance"],
        "Education": ["exam prep", "weekend study", "skill upgrade", "parenting", "commuting"],
        "Video": ["before sleep", "lunch break", "weekend binge", "traveling", "living room"],
        "Social": ["feeling lonely", "finding teammates", "weekend night", "sharing life", "networking"],
        "News": ["morning coffee", "commuting", "lunch break", "waiting in line", "restroom time"],
        "Tool": ["office work", "studying", "traveling", "shopping", "daily calculation"],
        "Travel": ["vacation planning", "business trip", "weekend getaway", "daily commute", "road trip"],
        "Shopping": ["salary day", "holiday sales", "grocery restock", "gift hunting", "window shopping"],
        "Finance": ["salary day", "market opening", "bill payment", "tax season", "daily expense check"],
        "Music": ["working out", "commuting", "focusing", "before sleep", "driving"],
        "Books": ["before sleep", "commuting", "quiet afternoon", "waiting", "vacation"]
    }
    
    tags_pool = {
        "Games": ["fun", "challenge", "kill time", "multiplayer", "puzzle", "strategy", "relax", "action"],
        "Utilities": ["clean", "speed up", "battery", "optimize", "efficiency", "storage", "security"],
        "Education": ["learning", "upskill", "courses", "exam", "reading", "skills", "kids"],
        "Video": ["movies", "short video", "drama", "anime", "entertainment", "live", "vlog"],
        "Social": ["chat", "dating", "moments", "voice call", "community", "friends", "strangers"],
        "News": ["hot topics", "local news", "world", "gossip", "reading", "updates", "finance news"],
        "Tool": ["scanner", "browser", "weather", "calendar", "efficiency", "calculator", "notes"],
        "Travel": ["booking", "flight", "hotel", "navigation", "taxi", "vacation", "guide"],
        "Shopping": ["discount", "ecommerce", "grocery", "fashion", "deals", "group buy", "makeup"],
        "Finance": ["banking", "invest", "loan", "payment", "stock", "wealth", "insurance"],
        "Music": ["listening", "karaoke", "podcast", "radio", "audio", "songs", "relax"],
        "Books": ["novel", "reading", "audiobook", "fiction", "comics", "literature", "fantasy"]
    }
    
    desc_templates = [
        "千万用户的首选{cat}应用！提供{feat}功能，让您在{scene}时享受极致体验。",
        "全新升级的{name}来了！专为{aud}打造，独创{feat}，解决您的核心痛点。",
        "还在为{scene}烦恼吗？试试这款{cat}神器，内置{feat}，轻松搞定！",
        "口碑爆棚的{name}，不仅支持{feat}，更是您{scene}的得力助手。",
        "懂你的{cat}平台，海量资源等你探索。无论是{scene}还是日常使用，都能满足{aud}的需求。"
    ]
    
    apps = []
    name_set = set()
    for i in range(count):
        cat = random.choice(categories)
        app_id = f"A{start_id + i:03d}"
        
        while True:
            app_name = f"{random.choice(prefixes)}{mid_words[cat]}{random.choice(suffixes)}"
            if app_name not in name_set:
                name_set.add(app_name)
                break
                
        # 随机抽取丰富特征
        ta = str(random.sample(target_audiences[cat], k=random.randint(1, 2)))
        cf = str(random.sample(core_features[cat], k=random.randint(2, 3)))
        sc = str(random.sample(scenarios[cat], k=random.randint(1, 2)))
        it = str(random.sample(tags_pool[cat], k=random.randint(2, 3)))
        
        # 随机生成描述
        template = random.choice(desc_templates)
        desc = template.format(
            cat=cat, 
            name=app_name, 
            feat=eval(cf)[0], # 取第一个核心功能填入文案
            scene=eval(sc)[0], # 取第一个场景填入文案
            aud=eval(ta)[0]    # 取第一个受众填入文案
        )
        
        apps.append({
            "app_id": app_id,
            "app_name": app_name,
            "description": desc,
            "category": cat,
            "target_audience": ta,
            "core_features": cf,
            "scenario": sc,
            "intent_tags": it
        })
    return apps

def expand_dataset():
    existing_csv = "data/app_features_enriched.csv"
    df_existing = pd.read_csv(existing_csv)
    
    # 核心操作：只保留前 50 条真实或高质量的数据，剔除之前的假数据
    df_existing = df_existing.head(50)
    
    current_count = len(df_existing)

    if current_count < 200:
        needed = 200 - current_count
        print(f"Current count is {current_count}. Generating {needed} highly diversified apps...")
        new_apps = generate_mock_apps(current_count + 1, needed)
        df_new = pd.DataFrame(new_apps)
        
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all.to_csv(existing_csv, index=False, encoding='utf-8-sig')
        print(f"Added {needed} apps. Total is now {len(df_all)}. Saved to {existing_csv}")

if __name__ == "__main__":
    expand_dataset()