import tkinter as tk
import NLPAug as npl
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import tkinter.ttk as ttk
import tkinter.filedialog as fd
import tkinter.messagebox as messagebox

#Build hàm


def NLPInsert(input):
    aug = nac.RandomCharAug(action="insert")
    augmented_text = aug.augment(input)
    return augmented_text

def NLPSplit(input):
    aug = naw.SplitAug()
    augmented_text = aug.augment(input)
    return augmented_text

def NLPSubstitute(input):
    aug = nac.RandomCharAug(action="substitute")
    augmented_text = aug.augment(input)
    return augmented_text

def NLPSwap(input):
    aug = nac.RandomCharAug(action="swap")
    augmented_text = aug.augment(input)
    return augmented_text

def NLPKeyboard(input):
    aug = nac.KeyboardAug()
    augmented_text = aug.augment(input)
    return augmented_text

def NLPBackTranslate(input):
    aug = naw.BackTranslationAug(from_model_name="facebook/wmt19-en-de", to_model_name="facebook/wmt19-de-en")
    augmented_text = aug.augment(input)
    return augmented_text

def NLPReserved(input):
    aug = naw.ReservedAug("substitute")
    augmented_text = aug.augment(input)
    return augmented_text

def NLPSynonym(input):
    aug = naw.ReservedAug("wordnet")
    augmented_text = aug.augment(input)
    return augmented_text




def NLPAug():
    input_text = text_box.get("1.0", tk.END).strip()
    method = combo.get() 
    
    if input_text:
        try:
            if method == "Thêm từ":
                augmented_text = NLPInsert(input_text)
            elif method == "Thay từ":
                augmented_text = NLPSubstitute(input_text)
            elif method == "Đổi từ":
                augmented_text = NLPSwap(input_text)
            elif method == "Back Translate":
                augmented_text = NLPBackTranslate(input_text)
            elif method == "Thay thực thể":
                augmented_text = NLPReserved(input_text)
            elif method == "Từ đồng nghĩa":
                augmented_text = NLPSynonym(input_text)
            elif method == "Tách từ":
                augmented_text = NLPSplit(input_text)
            else:
                augmented_text = "Vui lòng chọn phương thức tăng cường!"
        except Exception as e:
            augmented_text = f"Đã xảy ra lỗi: {str(e)}"
        
        # Hiển thị kết quả
        output_box.config(state="normal")
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, augmented_text)
        output_box.config(state="disabled")
    else:
        output_box.config(state="normal")
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Vui lòng nhập văn bản!")
        output_box.config(state="disabled")
def save_to_file():
    content = output_box.get("1.0", tk.END).strip()  # Lấy nội dung từ output_box
    if content:
        file = fd.asksaveasfile(
            mode="w", 
            defaultextension=".txt", 
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file:
            with open(file.name, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Thành công", "Văn bản đã được lưu!")  # Thông báo thành công
    else:
        messagebox.showwarning("Cảnh báo", "Không có nội dung để lưu!")  # Cảnh báo nếu không có nội dung


# Build giao diện

root = tk.Tk()
root.title("Tăng cường dữ liệu")
root.geometry("800x600")

label = tk.Label(root, text="Nhập văn bản của bạn: ", font=("Time New Roman", 14))
label.pack(pady=10)

text_box = tk.Text(root, height=10, width=50, font=("Time New Roman", 12))
text_box.pack(pady=10)

label = tk.Label(root, text="Văn bản tăng cường: ", font=("Time New Roman", 14))
label.pack(pady=10)

output_box = tk.Text(root, height=10, width=50, font=("Time New Roman", 12), state="disabled", bg="#f0f0f0")
output_box.pack(pady=10)

choice = ["Thêm từ", "Thay từ", "Đổi từ", "Back Translate", "Thay thực thể", "Từ đồng nghĩa", "Tách từ"]
combo = ttk.Combobox(root, values=choice, font=("Time New Roman", 12), state="readonly")
combo.set("Chọn phương thức")
combo.pack(pady=10)



button_frame = tk.Frame(root)
button_frame.pack(pady=10)

button1 = tk.Button(button_frame, text="Tăng cường", command=NLPAug, font=("Time New Roman", 12))
button2 = tk.Button(button_frame, text="Lưu", command=save_to_file, font=("Time New Roman", 12))

button1.pack(side=tk.LEFT, padx=10)
button2.pack(side=tk.LEFT, padx=10)


root.mainloop()