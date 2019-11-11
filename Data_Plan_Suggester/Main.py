import numpy as np
from os import path
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from keras.models import load_model

#1, 3
x = 1
y = 3
plans = {}
plans["mtn_monthly"] = np.array([1500, 2000, 3500, 6500, 11000, 25000])
plans_template = {}
plans_template["mtn_monthly"] = np.array([0, 1, 2, 3, 4, 5])

def suggest(*args):
    try:
        if(network.get() == ""):
            messagebox.showerror("Error", "Please select a network")
            return

        if(duration.get() == ""):
            messagebox.showerror("Error", "Please select a duration")
            return

        global usage
        value = float(usage.get())
        name = network.get() + "_" + duration.get()
        model_url = name + ".h5"

        if (not path.exists(model_url)):
            messagebox.showinfo("Error", "There is no model to load!")
            return

        model = load_model(model_url)

        plans_MB = plans[name]
        value /= 2500
        pred = model.predict([value])
        suggested_plan.set(plans_MB[pred.argmax()])

    except ValueError:
        messagebox.showinfo("Error", "Value error!")

def update_logo(*args):
    try:
        value = network.get()
        imageLabel['image'] = logos[value]
    except ValueError:
        pass

root = Tk()
root.title("Data Plan Suggester")
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

usage = DoubleVar()
suggested_plan = StringVar()
network = StringVar()
duration = StringVar()
logo_id = StringVar()

mtn_logo = PhotoImage(file='mtn.png')
glo_logo = PhotoImage(file='glo.png')
airtel_logo = PhotoImage(file='airtel.png')
nine_mobile_logo = PhotoImage(file='9mobile.png')

logos = {'mtn': mtn_logo, 'glo': glo_logo, 'airtel': airtel_logo, '9mobile': nine_mobile_logo}

imageLabel = ttk.Label(mainframe, textvariable=network)
imageLabel.grid(column=1, row=1)
usage_entry = ttk.Entry(mainframe, width=7, textvariable=usage)
usage_entry.grid(column=2, row=1, sticky=(W, E))
ttk.Label(mainframe, text='Use').grid(column=3, row=1, sticky=(W, E))
ttk.Label(mainframe, textvariable=suggested_plan).grid(column=2, row=2, sticky=(W, E))
ttk.Label(mainframe, text='You should use the').grid(column=1, row=2, sticky=(W, E))
ttk.Label(mainframe, text='MB data plan').grid(column=3, row=2, sticky=(W, E))
ttk.Button(mainframe, text="Suggest", command=suggest).grid(column=4, row=1, sticky=E)
ttk.Label(mainframe, text="MB").grid(column=3, row=1, sticky=W)
ttk.Label(mainframe, text="Duration").grid(column=2, row=3, sticky=W)
ttk.Label(mainframe, text="Network").grid(column=1, row=3, sticky=W)

ttk.Radiobutton(mainframe, text='MTN', variable=network, value='mtn', command=update_logo).grid(column=1, row=4, sticky=W)
ttk.Radiobutton(mainframe, text='Glo', variable=network, value='glo', command=update_logo).grid(column=1, row=5, sticky=W)
ttk.Radiobutton(mainframe, text='Airtel', variable=network, value='airtel', command=update_logo).grid(column=1, row=6, sticky=W)
ttk.Radiobutton(mainframe, text='Etisalat', variable=network, value='9mobile', command=update_logo).grid(column=1, row=7, sticky=W)

ttk.Radiobutton(mainframe, text='Daily', variable=duration, value='daily').grid(column=2, row=4, sticky=W)
ttk.Radiobutton(mainframe, text='Weekly', variable=duration, value='weekly').grid(column=2, row=5, sticky=W)
ttk.Radiobutton(mainframe, text='Monthly', variable=duration, value='monthly').grid(column=2, row=6, sticky=W)
    
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

usage_entry.focus()
root.bind('<Return>', suggest)
root.mainloop()
