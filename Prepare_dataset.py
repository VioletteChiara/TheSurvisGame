# Import Module
from tkinter import *



def rgb_to_hex(rgb):
    """Convert [R, G, B] list to '#RRGGBB' string"""
    return '#%02x%02x%02x' % tuple(rgb)



all_colors = [
    [255, 60, 71],     # tomato red
    [0, 191, 255],  # deep sky blue
    [147, 112, 219],  # medium purple
    [255, 215, 0],     # golden yellow
    [255, 99, 255],  # magenta
    [255, 165, 0],  # orange
    [64, 224, 208],  # turquoise
    [30, 144, 255],    # dodger blue
    [255, 105, 180],   # hot pink
    [255, 160, 122],   # light salmon

]

def prepare_dataset(points, previous_data=None):
    # create root window
    root = Tk()
    all_teams = []

    # root window title and dimension
    root.title("Team preparation")

    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=1000000)

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)

    Frame_teams=Frame(root)
    Frame_teams.grid(row=1, column=0, columnspan=2, sticky="nsew")
    Frame_teams.rowconfigure(0, weight=1)


    def add_team(name=None,team=None):
        all_teams.append(Team_Frame(Frame_teams, color=all_colors[len(all_teams)], points_init=points, previous_data=team, name=name))
        all_teams[-1].grid(row=0, column=len(all_teams)-1, sticky="nsew")
        Frame_teams.columnconfigure(len(all_teams)-1, weight=1)

    def del_team():
        all_teams[-1].grid_forget()
        del all_teams[-1]


    if not previous_data is None:
        for team in previous_data.keys():
            add_team(team,previous_data[team])


    final_dict = {}
    def begin_game(root):
        for Team in all_teams:
            Tname=Team.name.get()
            while Tname in final_dict.keys():
                Tname=Tname+" copy"
            final_dict[Tname]={}
            final_dict[Tname]["type"]=["H","C"][int(Team.HB.get())]
            final_dict[Tname]["lifespan_repro"] =Team.lifespan_repro.get()/20
            final_dict[Tname]["size_efficiency"] = Team.size_efficiency.get()
            final_dict[Tname]["eye"] = Team.eye.get()
            final_dict[Tname]["mandi"] = Team.mandi.get()
            final_dict[Tname]["speed"] = Team.legs.get()
            final_dict[Tname]["intelect"] = Team.intelect.get()/20
            final_dict[Tname]["protect"] = Team.protect.get()
            final_dict[Tname]["wings"]=1+Team.wings.get()//5
            final_dict[Tname]["col"] = [Team.color[2],Team.color[1],Team.color[0]]

        root.destroy()



    # all widgets will be here
    # Execute Tkinter

    if previous_data is None:
        B1 = Button(root, text="Add team", command=add_team).grid(row=0, column=0, sticky="nsew")
        B2 = Button(root, text="Remove team", command=del_team).grid(row=0, column=1, sticky="nsew")

    B3 = Button(root, text="Validate and build game!", command=lambda: begin_game(root)).grid(row=2, column=0,
                                                                                                   columnspan=2,
                                                                                                   sticky="nsew")

    root.mainloop()

    return(final_dict)



class Team_Frame(Frame):
    def __init__(self, parent, color, points_init=30,name=None, previous_data=None, **kwargs):
        Frame.__init__(self, parent, bd=5, **kwargs)
        dark_color=rgb_to_hex([max(0,v-75) for v in color])
        self.color=color
        self.col=rgb_to_hex(color)
        self.config(relief="ridge", highlightbackground=self.col, highlightthickness=4, highlightcolor=dark_color)

        self.points_init=points_init

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=20)

        cur_row=0
        #Team name
        Label(self,text="Team name: ",font=("Helvetica", 15)).grid(row=cur_row, column=0, sticky="ew")
        self.name=StringVar()
        self.entry_name=Entry(self, textvariable=self.name)
        self.entry_name.grid(row=cur_row, column=1,sticky="ew")
        self.rowconfigure(cur_row,weight=1)
        cur_row+=1




        #H vs C
        self.HB=BooleanVar()
        self.HB.set(False)
        Radiobutton(self,text="Herbivorous", value=0, variable=self.HB).grid(row=cur_row, column=0, sticky="nsew")
        Radiobutton(self, text="Carnivorous", value=1, variable=self.HB).grid(row=cur_row, column=1, sticky="nsew")
        cur_row += 1


        Frame_nopt=Frame(self, background=self.col)
        Frame_nopt.grid(row=cur_row, column=0, columnspan=2, sticky="nsew")
        cur_row += 1
        #Lifespan-reproduction
        Label(Frame_nopt,text="Reproduction", background=self.col).grid(row=1, column=0, sticky="e")
        self.lifespan_repro=IntVar()
        self.lifespan_repro.set(10)
        Scale(Frame_nopt, background=self.col, activebackground="red", sliderlength=15, sliderrelief="groove", highlightthickness=0, bd=0, relief="solid", variable=self.lifespan_repro,from_=0, to=20, orient=HORIZONTAL).grid(row=1, column=1,sticky="ew")
        Label(Frame_nopt, background=self.col, text="Lifespan").grid(row=1, column=2, sticky="w")
        self.rowconfigure(1, weight=1)

        #Size_efficiency
        Label(Frame_nopt, background=self.col, text="Food intake").grid(row=2, column=0, sticky="e")
        self.size_efficiency=IntVar()
        self.size_efficiency.set(10)
        Scale(Frame_nopt, background=self.col, activebackground="red", sliderlength=15, sliderrelief="groove", highlightthickness=0, bd=0, relief="flat", variable=self.size_efficiency,from_=0, to=20, orient=HORIZONTAL).grid(row=2, column=1,sticky="ew")
        Label(Frame_nopt, background=self.col, text="Size").grid(row=2, column=2, sticky="w")
        Frame_nopt.rowconfigure(2, weight=1)

        #Eye
        Label(self,text="Eye").grid(row=cur_row, column=0, sticky="e")
        self.eye=IntVar()
        self.eye.set(0)
        Scale(self, variable=self.eye,from_=0, to=20, orient=HORIZONTAL, command=lambda val, var=self.eye: self.recalculate(var, val)).grid(row=cur_row, column=1,sticky="ew")
        self.rowconfigure(cur_row, weight=1)
        cur_row += 1

        #Mandi
        Label(self,text="Mandibles").grid(row=cur_row, column=0, sticky="e")
        self.mandi=IntVar()
        self.mandi.set(0)
        Scale(self, variable=self.mandi,from_=0, to=20, orient=HORIZONTAL, command=lambda val, var=self.mandi: self.recalculate(var, val)).grid(row=cur_row, column=1,sticky="ew")
        self.rowconfigure(cur_row, weight=1)
        cur_row += 1

        #Legs
        Label(self,text="Legs").grid(row=cur_row, column=0, sticky="e")
        self.legs=IntVar()
        self.legs.set(0)
        Scale(self, variable=self.legs,from_=0, to=20, orient=HORIZONTAL, command=lambda val, var=self.legs: self.recalculate(var, val)).grid(row=cur_row, column=1,sticky="ew")
        self.rowconfigure(cur_row, weight=1)
        cur_row += 1

        #Protect
        Label(self,text="Protection").grid(row=cur_row, column=0, sticky="e")
        self.protect=IntVar()
        self.protect.set(0)
        Scale(self, variable=self.protect,from_=0, to=20, orient=HORIZONTAL, command=lambda val, var=self.protect: self.recalculate(var, val)).grid(row=cur_row, column=1,sticky="ew")
        self.rowconfigure(cur_row, weight=1)
        cur_row += 1


        #Intelect
        Label(self,text="Intelect").grid(row=cur_row, column=0, sticky="e")
        self.intelect=IntVar()
        self.intelect.set(0)
        Scale(self, variable=self.intelect,from_=0, to=20, resolution=1, orient=HORIZONTAL, command=lambda val, var=self.intelect: self.recalculate(var, val)).grid(row=cur_row, column=1,sticky="ew")
        self.rowconfigure(cur_row, weight=1)
        cur_row += 1


        #Wings
        Label(self,text="Wings").grid(row=cur_row, column=0, sticky="e")
        self.wings=IntVar()
        self.wings.set(0)
        Scale(self, variable=self.wings,from_=0, to=20, resolution=5, orient=HORIZONTAL, command=lambda val, var=self.wings: self.recalculate(var, val)).grid(row=cur_row, column=1,sticky="ew")
        self.rowconfigure(cur_row, weight=1)
        cur_row += 1

        self.points=IntVar()
        self.points.set(self.points_init)
        wrapper=Frame(self, highlightbackground=self.col, highlightthickness=5)
        wrapper.grid(row=100, column=0, columnspan=3, sticky="ns")
        self.rowconfigure(100, weight=1)
        Label(wrapper,text="Remaining points: ",font=("Helvetica", 15)).grid(row=100, column=0, sticky="e")
        Label(wrapper, textvariable=self.points,font=("Helvetica", 15)).grid(row=100, column=1, sticky="w")

        if not previous_data is None:
            if previous_data["type"]=="H":
                self.HB.set(0)
            else:
                self.HB.set(1)
            self.wings.set((previous_data["wings"]-1)*5)
            self.intelect.set(previous_data["intelect"]*20)
            self.protect.set(previous_data["protect"])
            self.legs.set(previous_data["speed"])
            self.mandi.set(previous_data["mandi"])
            self.eye.set(previous_data["eye"])
            self.size_efficiency.set(previous_data["size_efficiency"])
            self.lifespan_repro.set(previous_data["lifespan_repro"]*20)
            self.name.set(name)
            self.entry_name.config(state="disabled")
            self.recalculate(self.mandi,self.mandi.get())



    def recalculate(self, var, val):
        new_points=self.points_init-(self.eye.get())-(self.mandi.get())-(self.legs.get())-(self.protect.get())-(self.wings.get())-(self.intelect.get())
        if new_points>=0:
            self.points.set(new_points)
        else:
            if var!=self.wings:
                var.set(int(val)+new_points)
            else:
                var.set(int(val)+5*(new_points//5))

