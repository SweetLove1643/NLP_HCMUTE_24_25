import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

## Tăng cường dữ liệu bằng các phương pháp NLP
option_aug = {"Thêm từ",
           "Thay từ", 
           "Đổi vị trí", 
           "Back translate", 
           "Thay thực thể", 
           "Từ đồng nghĩa", 
           "Tách từ",
           "Lỗi keyboard"}
def NLPInsert(input):
    aug = nac.RandomCharAug(action="insert")
    augmented_text = aug.augment(input)
    return augmented_text[0]

def NLPSplit(input):
    aug = naw.SplitAug()
    augmented_text = aug.augment(input)
    return augmented_text[0]

def NLPSubstitute(input):
    aug = nac.RandomCharAug(action="substitute")
    augmented_text = aug.augment(input)
    return augmented_text[0]

def NLPSwap(input):
    aug = nac.RandomCharAug(action="swap")
    augmented_text = aug.augment(input)
    return augmented_text[0]

def NLPKeyboard(input):
    aug = nac.KeyboardAug()
    augmented_text = aug.augment(input)
    return augmented_text[0]

def NLPBackTranslate(input):
    aug = naw.BackTranslationAug(from_model_name="Helsinki-NLP/opus-mt-vi-en", to_model_name="Helsinki-NLP/opus-mt-en-vi")
    augmented_text = aug.augment(input)
    return augmented_text[0]

def NLPReserved(input):
    aug = naw.ReservedAug("substitute")
    augmented_text = aug.augment(input)
    return augmented_text[0]

def NLPSynonym(input):
    aug = naw.ReservedAug("wordnet")
    augmented_text = aug.augment(input)
    return augmented_text[0]