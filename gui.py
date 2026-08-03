import customtkinter as ctk
import os
import shutil
import threading
from tkinter import filedialog
from audio_separator.separator import Separator

# ================= 这里是调试信息 =================
print("⏳ 正在加载界面库，请稍候...")
# ================================================

# 设置外观
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AudioSeparatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 窗口设置
        self.title("音频分离工坊")
        self.geometry("700x600")
        
        # 数据变量
        self.input_path = ""
        self.output_path = os.path.join(os.getcwd(), "output")

        # ================= UI 布局 =================
        self.grid_columnconfigure(1, weight=1)

        # 1. 标题
        self.label_title = ctk.CTkLabel(self, text="AI 三轨极致分离系统", font=("Microsoft YaHei UI", 22, "bold"))
        self.label_title.grid(row=0, column=0, columnspan=3, pady=20)

        # 2. 输入文件
        self.btn_input = ctk.CTkButton(self, text="📂 选择音频文件", command=self.select_input_file)
        self.btn_input.grid(row=1, column=0, padx=20, pady=10)
        
        self.entry_input = ctk.CTkEntry(self, placeholder_text="请选择要处理的歌曲...")
        self.entry_input.grid(row=1, column=1, padx=(0, 20), pady=10, sticky="ew")

        # 3. 输出目录
        self.btn_output = ctk.CTkButton(self, text="📂 选择保存位置", fg_color="gray", command=self.select_output_folder)
        self.btn_output.grid(row=2, column=0, padx=20, pady=10)
        
        self.entry_output = ctk.CTkEntry(self, placeholder_text=self.output_path)
        self.entry_output.grid(row=2, column=1, padx=(0, 20), pady=10, sticky="ew")
        self.entry_output.insert(0, self.output_path)

        # 4. 开始按钮 (加大加粗)
        self.btn_start = ctk.CTkButton(self, text="🚀 开始分离 (去电音+去混响)", font=("Microsoft YaHei UI", 18), height=50, fg_color="#1f6aa5", hover_color="#144870", command=self.start_process_thread)
        self.btn_start.grid(row=3, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # 5. 日志框
        self.textbox_log = ctk.CTkTextbox(self, height=300, font=("Consolas", 12))
        self.textbox_log.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="nsew")
        self.grid_rowconfigure(4, weight=1)

        # 初始日志
        self.log("✨ 界面启动成功！等待任务中...")
        self.log("💡 提示：支持 GPU 加速，点击开始后请耐心等待模型加载。")

    def log(self, message):
        self.textbox_log.configure(state="normal")
        self.textbox_log.insert("end", message + "\n")
        self.textbox_log.see("end")
        self.textbox_log.configure(state="disabled")

    def select_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.ogg")])
        if file_path:
            self.input_path = file_path
            self.entry_input.delete(0, "end")
            self.entry_input.insert(0, file_path)
            self.log(f"✅ 已选中: {os.path.basename(file_path)}")

    def select_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_path = folder_path
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, folder_path)
            self.log(f"📂 保存路径: {folder_path}")

    def rename_helper(self, old_path, new_name):
        if os.path.exists(old_path):
            new_path = os.path.join(self.output_path, new_name)
            if os.path.exists(new_path): os.remove(new_path)
            os.rename(old_path, new_path)
            self.log(f"   -> 生成: {new_name}")
            return new_path
        return None

    def start_process_thread(self):
        if not self.input_path:
            self.log("⚠️ 请先选择一个音频文件！")
            return
        
        self.btn_start.configure(state="disabled", text="⏳ 正在全力处理中...", fg_color="#a63a3a")
        threading.Thread(target=self.run_separation_logic, daemon=True).start()

    def run_separation_logic(self):
        try:
            if not os.path.exists(self.output_path): os.makedirs(self.output_path)
            
            self.log("\n--- 正在初始化分离引擎 ---")
            sep = Separator(output_dir=self.output_path, output_format="flac")
            
            # Stage 1
            self.log("\n[1/3] 分离伴奏 (Roformer)...")
            sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
            step1_files = sep.separate(self.input_path)
            
            full_vocals_path = None
            for f in step1_files:
                path = os.path.join(self.output_path, f)
                if "(Instrumental)" in f:
                    self.rename_helper(path, "1_纯伴奏_Instrumental.flac")
                elif "(Vocals)" in f:
                    full_vocals_path = self.rename_helper(path, "temp_full_vocals.flac")

            # Stage 2
            lead_vocals_path = None
            if full_vocals_path:
                self.log("\n[2/3] 分离和声 (MDX-Karaoke)...")
                sep.load_model(model_filename="UVR_MDXNET_KARA_2.onnx")
                step2_files = sep.separate(full_vocals_path)
                
                for f in step2_files:
                    path = os.path.join(self.output_path, f)
                    if "(Instrumental)" in f:
                        self.rename_helper(path, "3_纯和声_Backing_Vocals.flac")
                    elif "(Vocals)" in f:
                        lead_vocals_path = self.rename_helper(path, "temp_lead_vocals.flac")

            # Stage 3
            if lead_vocals_path:
                self.log("\n[3/3] 去除混响 (De-Reverb)...")
                sep.load_model(model_filename="UVR-DeEcho-DeReverb.pth")
                step3_files = sep.separate(lead_vocals_path)

                for f in step3_files:
                    path = os.path.join(self.output_path, f)
                    if "(Instrumental)" in f:
                        self.rename_helper(path, "5_切除的混响.flac")
                    elif "(Vocals)" in f:
                        self.rename_helper(path, "2_纯主唱_干声.flac")

                if os.path.exists(full_vocals_path): os.remove(full_vocals_path)
                if os.path.exists(lead_vocals_path): os.remove(lead_vocals_path)

            self.log("\n🎉 === 处理完成！请查看文件夹 === \n")

        except Exception as e:
            self.log(f"\n❌ 出错啦: {str(e)}")
        
        finally:
            self.btn_start.configure(state="normal", text="🚀 开始分离 (Stage 1-3)", fg_color="#1f6aa5")

# ================= 重要：这就是你刚才缺少的启动代码 =================
if __name__ == "__main__":
    print("🚀 正在启动窗口循环...")
    app = AudioSeparatorApp()
    app.mainloop()
# ================================================================