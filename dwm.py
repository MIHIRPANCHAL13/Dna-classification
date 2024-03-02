import tkinter as tk
from tkinter import font as tkFont
import joblib
from PIL import Image, ImageTk


cv, classifier = joblib.load('model.pkl')

def getKmers(sequence, size=6):
    kmers = [sequence[i:i + size] for i in range(len(sequence) - size + 1)]
    return kmers

def predict_class():
    input_sequence = entry.get()
    input_kmers = getKmers(input_sequence, size=6)
    input_sentence = ' '.join(input_kmers)
    input_vector = cv.transform([input_sentence])
    predicted_class = classifier.predict(input_vector)
    result_label.config(text=f"Predicted Class: {predicted_class[0]}")


window = tk.Tk()
window.title("DNA Sequence Classifier")

font_style = tkFont.nametofont("TkDefaultFont")
font_style.configure(size=14)


label = tk.Label(window, text="Enter DNA Sequence:")
label.pack()

entry = tk.Entry(window, font=font_style)
entry.pack()

predict_button = tk.Button(window, text="Predict Class", command=predict_class, font=font_style)
predict_button.pack()

result_label = tk.Label(window, text="", font=font_style)
result_label.pack()

image = Image.open("E:\DWM\download.webp")
image = ImageTk.PhotoImage(image)
image_label = tk.Label(window, image=image)
image_label.pack()


window.mainloop()
