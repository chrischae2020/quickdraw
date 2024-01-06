from multiprocessing.connection import wait
from tkinter import *
import pickle
import os
import tkinter
from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk
from main import guess_was_right
from main import make_guess
import time


lastx, lasty = 0, 0			#to keep some initial point coordinates
prev_line=-1			#the last object operated on
color = "black"		
figure = "freehand"			
mode = "draw"				#main mode of the program ("draw","delete") 
counter_obj={"freehand":0}


def sign(number):
    if number>0:
        return 1;
    elif number<0:
        return -1;
    else:
        return 0;

    
def setFigure(newfigure):
    global figure, mode
    figure=newfigure
    mode="draw"
    #print("Mode: Draw ; Figure: "+figure)

def setSelection(newMode):
    global figure, mode, select_mode
    mode="delete"
    select_mode = newMode
    #print("Mode: Select ; Option: "+select_mode)
    

def addFree(event):													#create a new line, from last point to current, and tag it
    global lastx, lasty, counter_obj
    prefix='freehand_'
    canvas.create_line((lastx, lasty, event.x, event.y), fill=color, tags=(prefix+str(counter_obj["freehand"])), width="5p")
    draw.line((lastx, lasty, event.x, event.y), (0,0,0), width=5)
    lastx, lasty = event.x, event.y
    return prefix+str(counter_obj["freehand"])

def saveToDisk(): 	
    print("Processing...")
    global image, guess
    image = image.crop((200,0,900,500))		
    image.save("input.png")
    guess = ""
    guesses = make_guess()
    guess = guesses[-1]
    guess = str(guess)
    guess = guess[2:-1]
    id = canvas.create_text((570,495), text=f'Is it a {guess}?', 
                                tags=("result"),fill="black", font=("Helvetica", 35))
    def is_yes_lambda():
        return lambda x: handle_result("Woohoo")
    id = canvas.create_rectangle((505, 520, 560, 560), fill="black", tags=('result'))
    canvas.tag_bind(id, "<Button-1>", is_yes_lambda())
    id = canvas.create_text((520, 530), text="YES", tags=('result'), anchor=NW, fill="white")
    canvas.tag_bind(id, "<Button-1>", is_yes_lambda())
    def is_no_lambda():
        return lambda x: handle_result("Oh Noo")
    id = canvas.create_rectangle((575, 520, 630, 560), fill="black", tags=('result'))
    canvas.tag_bind(id, "<Button-1>", is_no_lambda())
    id = canvas.create_text((595, 530), text="NO", tags=('result'), anchor=NW, fill="white")
    canvas.tag_bind(id, "<Button-1>", is_no_lambda())
    
def handle_result(celebration):
    global guess, root
    delAllExceptButtons()
    if celebration == "Woohoo":     
        guess_was_right(guess)
        im = Image.open("generated.png").resize((440,440))
        img = ImageTk.PhotoImage(im)
        root.img = img

        #Add image to the Canvas Items
        canvas.create_image(350,30,anchor=NW,image=img, tags=("gen"))
        canvas.tag_raise("gen")
        canvas.tag_raise("buttons")
        id = canvas.create_text((575,450), text=f'{celebration}! Here is our AI generated image of a {guess}.',
                        tags=("congrats"),fill="black", font=("Helvetica", 35))	
        id = canvas.create_text((580,500), text=f'Play Again?',
                    tags=("congrats"),fill="black", font=("Helvetica", 35))	
        def is_yes_lambda():
            return lambda x: delAllExceptButtons()
        id = canvas.create_rectangle((505, 525, 560, 565), fill="black", tags=('congrats'))
        canvas.tag_bind(id, "<Button-1>", is_yes_lambda())
        id = canvas.create_text((520, 535), text="YES", tags=('congrats'), anchor=NW, fill="white")
        canvas.tag_bind(id, "<Button-1>", is_yes_lambda())	
        def is_no_lambda():
            return lambda x: root.quit()
        id = canvas.create_rectangle((575, 525, 630, 565), fill="black", tags=('congrats'))
        canvas.tag_bind(id, "<Button-1>", is_no_lambda())
        id = canvas.create_text((595, 535), text="NO", tags=('congrats'), anchor=NW, fill="white")
        canvas.tag_bind(id, "<Button-1>", is_no_lambda())
    else:
        id = canvas.create_text((570,280), text=f'{celebration}! No one is perfect.',
            tags=("congrats"),fill="black", font=("Helvetica", 35))
        id = canvas.create_text((580,320), text=f'Play Again?',
                    tags=("congrats"),fill="black", font=("Helvetica", 35))		
        def is_yes_lambda():
            return lambda x: delAllExceptButtons()
        id = canvas.create_rectangle((505, 350, 560, 390), fill="black", tags=('congrats'))
        canvas.tag_bind(id, "<Button-1>", is_yes_lambda())
        id = canvas.create_text((520, 360), text="YES", tags=('congrats'), anchor=NW, fill="white")
        canvas.tag_bind(id, "<Button-1>", is_yes_lambda())	
        def is_no_lambda():
            return lambda x: root.quit()
        id = canvas.create_rectangle((575, 350, 630, 390), fill="black", tags=('congrats'))
        canvas.tag_bind(id, "<Button-1>", is_no_lambda())
        id = canvas.create_text((595, 360), text="NO", tags=('congrats'), anchor=NW, fill="white")
        canvas.tag_bind(id, "<Button-1>", is_no_lambda())	
	
def delAllExceptButtons():								#deletes everything on screen except buttons
    global image, draw
    items = canvas.find_all()
    image = Image.new("RGB", (900, 600), (255,255,255))
    draw = ImageDraw.Draw(image)
    for curr_item in items:
        tags_list=canvas.itemcget(curr_item, "tags").split(" ")
        flag=False
        for curr_tag in tags_list:						#check all tags for button object
            if curr_tag.find("buttons")!=-1:
                flag=True
        if not flag:									#if its not a buttons object
            canvas.delete(curr_item)
    
def left_click(event):
    global lastx, lasty, prev_line
    if mode == "draw":
        lastx, lasty = event.x, event.y
        prev_line=-1
    elif mode == "delete":
        delAllExceptButtons()
        setFigure("freehand")


def mouse_move(event):
    if mode=="draw": 
        if figure=="freehand":
            addFree(event);
        else:
            print("ERROR")
    else:
        print("ERROR")
    canvas.tag_raise("buttons") 								


root = Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.geometry("1200x600")

canvas = Canvas(root, background="white")
canvas.grid(column=0, row=0, sticky=(N, W, E, S))
canvas.bind("<Button-1>", left_click)
canvas.bind("<B1-Motion>", mouse_move)

image = Image.new("RGB", (900, 600), (255,255,255))
draw = ImageDraw.Draw(image)

categories=["bed", "chair", "table", "bench", "door", "TV", "sink", "fan", "toilet", "power outlet"]
counter=10
id = canvas.create_text((15, counter), text="List of Categories:", tags=('buttons'), anchor=NW, fill="black")
counter = 40
for curr in categories:
    id = canvas.create_text((15, counter+2), text=f'  - {curr}', tags=('buttons'), anchor=NW, fill="black")
    counter = counter + 25;


def selection_lambda(var):
    return lambda x: setSelection(var)
id = canvas.create_rectangle((counter+100, 10, counter+160, 30), fill="black", tags=('buttons'))
canvas.tag_bind(id, "<Button-1>", selection_lambda("delete"))
id = canvas.create_text((counter+105, 10), text="Delete", tags=('buttons'), anchor=NW, fill="white")
canvas.tag_bind(id, "<Button-1>", selection_lambda("delete"))
counter = counter + 75;


def saving_lambda(var):
    return lambda x: saveToDisk()
id = canvas.create_rectangle((counter+200, 10, counter+260, 30), fill="black", tags=('buttons'))
canvas.tag_bind(id, "<Button-1>", saving_lambda("save"))
id = canvas.create_text((counter+205, 10), text="Submit", tags=('buttons'), anchor=NW, fill="white")
canvas.tag_bind(id, "<Button-1>", saving_lambda("save"))
counter = counter + 75;

def run():
    root.mainloop()
    exit()
run()