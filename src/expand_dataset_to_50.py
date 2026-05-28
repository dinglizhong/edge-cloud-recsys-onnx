import pandas as pd
import os

def expand_dataset():
    existing_csv = "data/app_features_enriched.csv"
    if not os.path.exists(existing_csv):
        print(f"Error: {existing_csv} not found.")
        return

    df_existing = pd.read_csv(existing_csv)
    print(f"Current dataset size: {len(df_existing)} rows.")

    # 42 new apps with pre-extracted simulated LLM features to save API costs and time
    apps_data = [
        ("A009", "微信", "国民级社交软件，支持语音视频聊天、朋友圈分享生活，微信支付便捷买单。", "Social", "['all ages']", "['instant messaging', 'video call', 'moments', 'mobile payment']", "['daily communication', 'social networking', 'payment']", "['chat', 'social', 'pay']"),
        ("A010", "QQ", "年轻人都在用的社交平台，支持多人视频、文件传输、个性装扮。", "Social", "['youth', 'students']", "['group video', 'file transfer', 'customization']", "['school communication', 'gaming social', 'file sharing']", "['chat', 'social', 'youth']"),
        ("A011", "微博", "随时随地发现新鲜事！汇聚娱乐八卦、社会热点，明星动态一手掌握。", "News", "['young adults', 'fans']", "['trending topics', 'celebrity news', 'social media']", "['browsing news', 'following celebrities', 'entertainment']", "['news', 'gossip', 'entertainment']"),
        ("A012", "哔哩哔哩", "国内知名的视频弹幕社区，涵盖动漫、番剧、国创、游戏视频。", "Video", "['Gen Z', 'ACG fans']", "['danmaku', 'anime', 'user generated content']", "['watching anime', 'learning', 'entertainment']", "['video', 'anime', 'community']"),
        ("A013", "快手", "国民短视频社区，记录真实生活，看搞笑段子、直播带货。", "Video", "['general public']", "['short video', 'live streaming', 'e-commerce']", "['passing time', 'entertainment', 'shopping']", "['short video', 'live', 'fun']"),
        ("A014", "爱奇艺", "海量正版高清影视剧，独播综艺、热播电视剧第一时间追。", "Video", "['drama fans', 'movie lovers']", "['movies', 'TV series', 'variety shows']", "['watching movies', 'binge-watching TV shows']", "['video', 'movies', 'entertainment']"),
        ("A015", "腾讯视频", "海量独播内容，热播剧集、热门综艺、院线大片应有尽有。", "Video", "['drama fans', 'movie lovers']", "['movies', 'TV series', 'exclusive content']", "['watching movies', 'binge-watching TV shows']", "['video', 'movies', 'entertainment']"),
        ("A016", "优酷", "海量高清视频在线观看，提供电视剧、电影、动漫、综艺节目。", "Video", "['drama fans', 'movie lovers']", "['movies', 'TV series', 'anime']", "['watching movies', 'binge-watching TV shows']", "['video', 'movies', 'entertainment']"),
        ("A017", "芒果TV", "湖南卫视独播综艺、自制网剧平台，追剧必备神器。", "Video", "['young females', 'variety show fans']", "['exclusive variety shows', 'web dramas']", "['watching variety shows', 'entertainment']", "['video', 'variety', 'entertainment']"),
        ("A018", "知乎", "高质量问答社区，专业人士分享知识、经验和见解。", "Community", "['professionals', 'students']", "['Q&A', 'articles', 'knowledge sharing']", "['learning', 'problem solving', 'reading']", "['knowledge', 'Q&A', 'learning']"),
        ("A019", "今日头条", "基于个性化推荐引擎的资讯聚合平台，懂你感兴趣的新闻。", "News", "['general public', 'news readers']", "['personalized news', 'articles', 'videos']", "['reading news', 'passing time']", "['news', 'information', 'reading']"),
        ("A020", "百度", "全球最大的中文搜索引擎，搜你想搜，看你想看。", "Search", "['all users']", "['web search', 'news feed', 'voice search']", "['searching information', 'reading news']", "['search', 'information', 'tool']"),
        ("A021", "夸克", "极速极简的智能浏览器，自带强大的网盘和扫描功能。", "Tool", "['youth', 'students', 'professionals']", "['fast browsing', 'cloud storage', 'scanner']", "['web browsing', 'file storage', 'scanning']", "['browser', 'tool', 'efficiency']"),
        ("A022", "高德地图", "精准的手机地图导航软件，提供打车、公交、驾车路线规划。", "Navigation", "['drivers', 'travelers', 'commuters']", "['navigation', 'ride-hailing', 'public transit']", "['driving', 'commuting', 'traveling']", "['map', 'navigation', 'travel']"),
        ("A023", "百度地图", "智能地图导航，提供路况播报、全景地图、语音导航。", "Navigation", "['drivers', 'travelers', 'commuters']", "['navigation', 'real-time traffic', 'street view']", "['driving', 'commuting', 'traveling']", "['map', 'navigation', 'travel']"),
        ("A024", "滴滴出行", "一站式出行服务平台，提供快车、专车、代驾等出行服务。", "Travel", "['commuters', 'travelers']", "['ride-hailing', 'designated driver', 'taxi']", "['commuting', 'traveling', 'going out']", "['taxi', 'travel', 'ride']"),
        ("A025", "携程旅行", "提供酒店、机票、火车票、景点门票等一站式旅游预订服务。", "Travel", "['travelers', 'business travelers']", "['hotel booking', 'flight booking', 'train tickets']", "['travel planning', 'business trips']", "['travel', 'hotel', 'flight']"),
        ("A026", "去哪儿旅行", "特价机票、酒店预订平台，让你的旅行更省钱。", "Travel", "['budget travelers', 'students']", "['discount flights', 'hotel booking', 'vacation packages']", "['travel planning', 'budget travel']", "['travel', 'discount', 'booking']"),
        ("A027", "飞猪旅行", "阿里旗下旅行品牌，提供高性价比的国内外旅游服务。", "Travel", "['travelers', 'youth']", "['hotel booking', 'flight booking', 'tour packages']", "['travel planning', 'vacations']", "['travel', 'booking', 'vacation']"),
        ("A028", "铁路12306", "中国铁路官方购票软件，支持车票查询、预订、退改签。", "Travel", "['travelers', 'commuters']", "['train ticket booking', 'ticket refund/change']", "['taking train', 'travel planning']", "['train', 'ticket', 'travel']"),
        ("A029", "淘宝", "亚洲较大的网上交易平台，海量商品，淘你喜欢。", "Shopping", "['all users']", "['online shopping', 'live commerce', 'customer reviews']", "['shopping', 'browsing products']", "['shopping', 'ecommerce', 'buy']"),
        ("A030", "京东", "正品低价，极速物流。提供家电、数码、服饰等全品类网购。", "Shopping", "['professionals', 'families']", "['fast delivery', 'authentic products', 'electronics']", "['buying electronics', 'urgent shopping']", "['shopping', 'ecommerce', 'fast delivery']"),
        ("A031", "拼多多", "新电商开创者，拼着买，更便宜。百亿补贴正品保障。", "Shopping", "['budget shoppers', 'families']", "['group buying', 'huge discounts', 'subsidies']", "['budget shopping', 'buying groceries']", "['shopping', 'discount', 'group buy']"),
        ("A032", "闲鱼", "阿里巴巴旗下闲置交易平台，让你的闲置物品游起来。", "Shopping", "['youth', 'budget shoppers']", "['second-hand trading', 'community', 'auctions']", "['selling used items', 'buying cheap items']", "['second-hand', 'shopping', 'trade']"),
        ("A033", "美团", "吃喝玩乐全都有，外卖、电影票、酒店预订一站式生活服务。", "Local Life", "['all users']", "['food delivery', 'movie tickets', 'local services']", "['ordering food', 'going out', 'entertainment']", "['food', 'delivery', 'local']"),
        ("A034", "饿了么", "专业的外卖订餐平台，准时必达，超时赔付。", "Local Life", "['all users', 'office workers']", "['food delivery', 'grocery delivery']", "['ordering meals', 'buying snacks']", "['food', 'delivery', 'takeout']"),
        ("A035", "大众点评", "发现品质生活，查看网友真实评价，找高分餐厅。", "Local Life", "['foodies', 'young adults']", "['restaurant reviews', 'deals', 'local exploration']", "['finding restaurants', 'reading reviews']", "['food', 'reviews', 'local']"),
        ("A036", "58同城", "找工作、找房子、找二手，同城生活服务大全。", "Local Life", "['job seekers', 'renters']", "['job hunting', 'apartment renting', 'second-hand']", "['moving', 'finding jobs', 'local services']", "['local', 'housing', 'jobs']"),
        ("A037", "支付宝", "国民级生活服务平台，提供支付、理财、生活缴费等功能。", "Finance", "['all users']", "['mobile payment', 'wealth management', 'utility bills']", "['paying bills', 'managing money', 'shopping']", "['payment', 'finance', 'tool']"),
        ("A038", "云闪付", "银联官方App，支持各种银行卡管理，支付优惠多。", "Finance", "['bank card users']", "['card management', 'mobile payment', 'discounts']", "['paying in stores', 'managing cards']", "['payment', 'finance', 'bank']"),
        ("A039", "招商银行", "招商银行官方手机银行，提供转账、理财、信用卡服务。", "Finance", "['CMB customers', 'professionals']", "['transfer', 'wealth management', 'credit card']", "['managing bank account', 'investing']", "['bank', 'finance', 'money']"),
        ("A040", "喜马拉雅", "海量有声书、相声评书、新闻播客，听你想听。", "Audio", "['commuters', 'audiobook lovers']", "['audiobooks', 'podcasts', 'radio shows']", "['commuting', 'before sleep', 'relaxing']", "['audio', 'listening', 'podcast']"),
        ("A041", "蜻蜓FM", "汇聚广播电台、有声小说、儿童故事，随时随地听世界。", "Audio", "['radio listeners', 'parents']", "['radio stations', 'audiobooks', 'kids stories']", "['driving', 'relaxing', 'parenting']", "['audio', 'radio', 'listening']"),
        ("A042", "酷狗音乐", "海量正版曲库，蝰蛇音效，随时随地享受好音乐。", "Music", "['music lovers', 'general public']", "['music streaming', 'sound effects', 'KTV']", "['listening to music', 'relaxing', 'karaoke']", "['music', 'audio', 'entertainment']"),
        ("A043", "全民K歌", "腾讯出品的K歌软件，智能修音，和好友一起在线欢唱。", "Music", "['singing lovers', 'youth']", "['karaoke', 'voice tuning', 'social singing']", "['singing', 'party', 'entertainment']", "['music', 'singing', 'social']"),
        ("A044", "番茄免费小说", "海量正版小说免费看，都市、玄幻、言情全都有。", "Books", "['fiction readers']", "['free novels', 'audio reading', 'huge library']", "['reading', 'passing time', 'relaxing']", "['reading', 'books', 'novel']"),
        ("A045", "七猫免费小说", "免费看书100年，精品小说大全，阅读还能赚零花钱。", "Books", "['fiction readers', 'budget users']", "['free novels', 'earn money reading', 'huge library']", "['reading', 'passing time', 'earning rewards']", "['reading', 'books', 'free']"),
        ("A046", "百度网盘", "超大空间的云存储产品，支持文件备份、分享、在线预览。", "Tool", "['students', 'professionals']", "['cloud storage', 'file backup', 'file sharing']", "['storing files', 'sharing documents', 'working']", "['tool', 'storage', 'productivity']"),
        ("A047", "腾讯会议", "高清流畅的云视频会议软件，支持屏幕共享、在线协作。", "Productivity", "['professionals', 'students']", "['video conference', 'screen sharing', 'collaboration']", "['remote work', 'online classes', 'meetings']", "['work', 'meeting', 'productivity']"),
        ("A048", "钉钉", "阿里出品的企业级智能移动办公平台，提升沟通协同效率。", "Productivity", "['professionals', 'companies']", "['team communication', 'attendance tracking', 'approval workflows']", "['working', 'company management']", "['work', 'office', 'productivity']"),
        ("A049", "扫描全能王", "手机变身扫描仪，高清扫描文档，文字识别提取。", "Tool", "['professionals', 'students']", "['document scanning', 'OCR text recognition', 'PDF export']", "['scanning documents', 'studying', 'working']", "['tool', 'scanner', 'productivity']"),
        ("A050", "剪映", "全能好用的视频编辑工具，海量模板，轻松剪出大片。", "Tool", "['video creators', 'vloggers']", "['video editing', 'templates', 'effects and filters']", "['editing videos', 'creating vlogs', 'TikTok creation']", "['tool', 'video editing', 'creation']")
    ]

    new_rows = []
    for app_id, app_name, desc, category, ta, cf, sc, it in apps_data:
        new_rows.append({
            "app_id": app_id,
            "app_name": app_name,
            "description": desc,
            "category": category,
            "target_audience": ta,
            "core_features": cf,
            "scenario": sc,
            "intent_tags": it
        })

    df_new = pd.DataFrame(new_rows)
    
    # 过滤掉可能已经存在的重复数据
    existing_ids = set(df_existing['app_id'].tolist())
    df_new = df_new[~df_new['app_id'].isin(existing_ids)]
    
    if len(df_new) > 0:
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all.to_csv(existing_csv, index=False, encoding='utf-8-sig')
        print(f"Successfully appended {len(df_new)} new apps.")
        print(f"New dataset size: {len(df_all)} rows. Saved to {existing_csv}")
    else:
        print("Dataset already contains 50 rows, no new apps added.")

if __name__ == "__main__":
    expand_dataset()
