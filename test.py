from happytransformer import HappyTextToText
from happytransformer import TTSettings
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import string

# modif

from flask import Flask, render_template, url_for, request

app = Flask(__name__)

 

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
    output = request.form.to_dict()
    print(output)
    raw_text = output["inputText"]
    input_text = raw_text

    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction") #Model

    # raw_text = "whle you make the first two of the contract paymens as agred. you has refused to make the fincel $3 600 payment" #sample text
    # print(raw_text)

    unpunc_model = re.compile("["+ re.escape(string.punctuation) + "]") 

    unpunc_text = unpunc_model.sub("", raw_text) 

    tokenize_text = word_tokenize(unpunc_text)

    #raw_text = "whle you made the first two of the contract paymens as agred you have refused to make the fincel $3 600 payment"

    processed_text = [] 

    for i in range(0, len(tokenize_text)):
      modeled_text= TextBlob(tokenize_text[i]) 
      correct_word= modeled_text.correct() 
      processed_text.append(correct_word) 

 
    ripe_text= " ".join(map(str, processed_text)) 

    # print(ripe_text)


    text_tokenize = sent_tokenize(ripe_text)

    args = TTSettings(num_beams=5, min_length=1)

    processed_text= []

    for i in range(0, len(text_tokenize)):
      modeled_text= happy_tt.generate_text(text_tokenize[i], args=args) 
      processed_text.append(modeled_text.text)

    ripe_text= " ".join(map(str, processed_text)) 
    # print(ripe_text)

    return render_template('index.html', ripe_text = ripe_text, input_text=raw_text)
    

# end


if __name__ == "__main__":
    app.run(debug=True)






