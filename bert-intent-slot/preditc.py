from detector import JointIntentSlotDetector
import time

start1_time = time.perf_counter()
model = JointIntentSlotDetector.from_pretrained(
    model_path='./save_model/bert-base-chinese',
    tokenizer_path='./save_model/bert-base-chinese',
    intent_label_path='./data/SMP2019/intent_labels.txt',
    slot_label_path='./data/SMP2019/slot_labels.txt'
)
start2_time = time.perf_counter()
all_text = ['定位我现在的位置', "现在几点了", "2013年亚洲冠军联赛恒广州恒大比赛时间。", "帮我查一下赣州到厦门的汽车", "导航到望江西路上去", "把张玉娟的手机号码发送给吴伟", "打电话给xxx", "经XXX的电话号码发给lc"
            "发信息给盛吉", "将你在哪发送给纲吉", "发信息给老妈说我在吃饭", "我要听稻香", "访问浏览器", "中国制用英文怎么说"]
for i in all_text:
    print(model.detect(i))
end_time = time.perf_counter()
time1 = (end_time - start1_time) / 3600
time2 = (end_time - start2_time) / 3600
print("所有检测时间（包括加载模型）：", time1, "s", "除去模型加载时间：", time2, "s",
      "总预测数据量：", len(all_text), "平均预测一条的时间（除去加载模型）：", time2 / len(all_text), "s/条")
