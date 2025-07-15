# Chatbot geliştirmek ve bunu bir arayüz ile kullanıcıya sunmak.


# Veri -> İşin en zor kısmı.
# LLM = Large Language Model


# 1. teknik -> Herhangi bir modeli olduğu gibi kullanmak
# 2. teknik -> Modeli alıp özel veriyle donatma -> Parametre sayısı


# Transfer-Learning -> Daha önceden büyük çaplı bir veriyle eğitilmiş bir modeli alıp kendi verimizle çalışabilecek duruma getirmek.
# Derin Öğrenme => Sıfırdan bir beyin eğitmek.
# Transfer Learning -> Benzer verilerle eğitilmiş bir beyni kendi verimize adapte etmek..


from transformers import AutoTokenizer, TFAutoModelForCausalLM
import tensorflow as tf

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForCausalLM.from_pretrained(model_name)

# neden pytorch istedi?
#pipeline -> pytorch ile çalışıyor.
##generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
##response = generator(prompt, max_length=100, pad_token_id=tokenizer.eos_token_id)
#print(response)
prompt = (
    "User:	Does money buy happiness?\n"
    "Bot:	Depends how much money you spend on it .\n"
    "User:	What is the best way to buy happiness ?\n"
    "Bot:	You just have to be a millionaire by your early 20s, then you can be happy .\n"
    "User:	This is so difficult !\n"
)

input_ids = tokenizer.encode(prompt, return_tensors="tf")

output = model.generate(input_ids, 
                        max_length=100, 
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.8,
                        top_k=100,
                        top_p=0.50
                        )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)




