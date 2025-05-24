from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
model_name = "/mnt/data/chatglm3-6b" # 本地路径
prompt = "请回答如下三个问题：1.请说说冬天能穿多少穿多少和夏天能穿多少穿多少这两句话的区别？2.请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。3.明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？"
tokenizer = AutoTokenizer.from_pretrained(
model_name,
trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
model_name,
trust_remote_code=True,
torch_dtype="auto" # 自动选择 float32/float16（根据模型配置）
).eval()
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)