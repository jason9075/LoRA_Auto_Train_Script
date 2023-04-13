from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

# 設定要載入的事件檔案路徑
event_file = "./log/20230330060418/network_train/events.out.tfevents.1680127519.jason9075-popos.134289.0"

# 載入事件檔案
loader = EventFileLoader(event_file)

# 讀取每個事件
for event in loader.Load():
    # 判斷事件類型
    if event.HasField("summary"):
        # 輸出事件的摘要內容
        print(event.summary.value)
    """ elif event.HasField("step"): """
    """     # 輸出事件的步數 """
    """     print(event.step) """
