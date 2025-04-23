import nlpaug.augmenter.char as nac

# Tạo augmenter cho thao tác hoán đổi ký tự
char_aug = nac.RandomCharAug(action="swap")

text = "Hello, how are you?"
augmented_text = char_aug.augment(text)

print("Original:", text)
print("Augmented:", augmented_text)

#Thay từ, đổi vị trí, thay thực thể, dịch ngược, từ đồng nghĩa,...