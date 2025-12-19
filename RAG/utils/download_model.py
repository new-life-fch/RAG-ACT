import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 从环境变量获取 Hugging Face Token
HF_TOKEN = os.getenv('HF_TOKEN')
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN未设置，请在.env文件中设置HF_TOKEN")

from huggingface_hub import snapshot_download, login

# 登录 Hugging Face
try:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ 已使用 HF_TOKEN 登录")
except Exception as e:
    print(f"❌ 登录失败: {e}")



# 模型名称
repo_id = "lmsys/vicuna-7b-v1.5"

# 本地保存路径
local_dir = "/root/shared-nvme/RAG-llm/models/vicuna-7b-v1.5"    
os.makedirs(local_dir, exist_ok=True)

print(f"⬇️ 正在从官方 Hugging Face 下载模型 '{repo_id}' 到 '{local_dir}'...")

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 避免符号链接问题（服务器环境推荐）
        force_download=False,  # 已下载的文件不重复下载（需重下则改为True）
        resume_download=True,  # 支持断点续传（网络中断后可继续）
        # ignore_patterns=["*.bin", "*.pth"],  # 过滤掉 .bin 和 .pth 文件
        max_workers=8,  # 多线程下载（加速）
    )
    print("✅ 模型下载成功！")
except Exception as e:
    print(f"❌ 模型下载失败: {e}")
    print("可能的原因：")
    print("1. 你的 HF_TOKEN 没有该模型的访问权限，需要去模型页面点 Request access")
    print("2. 网络无法访问 huggingface.co")
    print("3. repo_id 拼写错误（请检查是否正确）")