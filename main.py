import random
import os
import cv2
import numpy as np
import Class_Herbivorous
import math
import time
import Prepare_dataset
import sys
import keyboard
import Do_pie

Color_fond = [80,208,146]

import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()-75
root.destroy()

def resize_img(img):
    h, w = img.shape[:2]
    scale = min(screen_width / w, screen_height / h)

    # Resize image
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return(resized_img)

def rankdata(a, method='average'):
    """
    A pure NumPy implementation of scipy.stats.rankdata.
    Supported methods: 'average', 'min', 'max', 'dense', 'ordinal'
    """
    a = np.ravel(np.asarray(a))
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    arr = a[sorter]

    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense_ranks = np.cumsum(obs)
    if method == 'dense':
        return dense_ranks[inv]

    count = np.r_[np.nonzero(obs)[0], len(a)]
    ranks = np.zeros(len(a), dtype=float)
    for i in range(len(count) - 1):
        start, end = count[i], count[i + 1]
        if method == 'average':
            ranks[start:end] = 0.5 * (start + end - 1)
        elif method == 'min':
            ranks[start:end] = start
        elif method == 'max':
            ranks[start:end] = end - 1
        elif method == 'ordinal':
            ranks[start:end] = np.arange(start, end)
        else:
            raise ValueError(f"Unknown method: {method}")

    return ranks[inv] + 1.0  # scipy ranks start at 1



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=Color_fond)
  return result



#Time is in sec:
dt=1/20



#The environment is a grid:
#Environment size:
height=Class_Herbivorous.height
width=Class_Herbivorous.width
Environment = np.random.randint(0, 256, (height, width), dtype=np.uint16)
Environment=Environment.astype(np.float32)
#Environment.fill(255)
Environment[0:height,0]=0
Environment[0:height,width-1]=0
Environment[height-1,0:width]=0
Environment[0,0:width]=0

growth_rate=0.03 #En % par sec

#For visual representation
block_size = 75



max_points=60
min_points=20
points_per_round=range(min_points,max_points+1,round((max_points-min_points)/4))

All_illus=[]
All_nb_alive=[]
All_Nb_ever_living=[]
All_offspring=[]
All_All_food_eaten=[]


for round_G in range(5):
    if round_G==0:
        all_teams, duration = Prepare_dataset.prepare_dataset(points=points_per_round[round_G], previous_data=None, width=screen_width-100)
    else:
        all_teams, duration=Prepare_dataset.prepare_dataset(points=points_per_round[round_G],previous_data=all_teams, width=screen_width-100)

    death_type={Team:{} for Team in all_teams.keys()}
    total_nb_data={Team:0 for Team in all_teams.keys()}
    total_feeding={Team:0 for Team in all_teams.keys()}
    total_repro = {Team:[] for Team in all_teams.keys()}

    herbivores = {}
    for Team in all_teams.keys():
        if all_teams[Team]["type"]=="H":
            nb=25
            total_nb_data[Team]+=25
        else:
            nb=10
            total_nb_data[Team] += 10

        herbivores[Team]=[Class_Herbivorous.Herbivorous(height,width, Team,**all_teams[Team], age=random.random()*5) for _ in range(nb)]

    Body=cv2.imread(resource_path(os.path.join("Images","Body.png")))
    Zebra=cv2.imread(resource_path(os.path.join("Images","zebra.png")))


    Size_show=300
    Espacio_texto=50
    Illu_to_show=np.zeros([Size_show+Espacio_texto,Size_show*len(all_teams.keys()),3], dtype=np.uint8)
    Illu_to_show[:]=Color_fond
    Team_count=0
    for Team in all_teams.keys():
        #Legs
        if all_teams[Team]["speed"]>0:
            file=resource_path(os.path.join("Images", "legs"+str(all_teams[Team]["speed"])+".png"))
            Illu = cv2.imread(file)
            Illu=Illu[1:2344, 1:2344]
        else:
            Illu=np.zeros([2343,2343,3], dtype=np.uint8)
            Illu[:]=Color_fond

        #Mandibles
        if all_teams[Team]["mandi"] > 0:
            file=resource_path(os.path.join("Images", "mandibles"+str(all_teams[Team]["mandi"])+".png"))
            mandi_img = cv2.imread(file)
            mandi_img=mandi_img[1:2344, 1:2344]
            Illu[np.any(mandi_img != Color_fond, axis=-1)] = mandi_img[np.any(mandi_img != Color_fond, axis=-1)]



        body_img=Body[1:2344, 1:2344].copy()
        Illu[np.any(body_img != Color_fond, axis=-1)] = body_img[np.any(body_img != Color_fond, axis=-1)]

        if all_teams[Team]["type"]=="C":
            zebra_img=Zebra[1:2344, 1:2344].copy()
            zebra_img=rotate_image(zebra_img,random.random()*360)
            mask = np.any((zebra_img != Color_fond) & (body_img != Color_fond), axis=-1)
            Illu[mask] = zebra_img[mask]




        if all_teams[Team]["eye"] > 0:
            file=resource_path(os.path.join("Images", "eye"+str(all_teams[Team]["eye"])+".png"))
            eye_img = cv2.imread(file)
            eye_img = eye_img[1:2344, 1:2344]
            eye_img2 = cv2.flip(eye_img, 0)

            Illu[np.any(eye_img != Color_fond, axis=-1)] = eye_img[np.any(eye_img != Color_fond, axis=-1)]
            Illu[np.any(eye_img2 != Color_fond, axis=-1)] = eye_img2[np.any(eye_img2 != Color_fond, axis=-1)]

        if all_teams[Team]["wings"]>1:
            file=resource_path(os.path.join("Images", "wings"+str(all_teams[Team]["wings"])+".png"))
            wing_img = cv2.imread(file)
            wing_img = wing_img[1:2344, 1:2344]
            Illu[np.any(wing_img != Color_fond, axis=-1)] = wing_img[np.any(wing_img != Color_fond, axis=-1)]

        if all_teams[Team]["protect"] > 0:
            file=resource_path(os.path.join("Images", "protection"+str(all_teams[Team]["protect"])+".png"))
            protec_img = cv2.imread(file)
            protec_img = protec_img[1:2344, 1:2344]
            Illu[np.any(protec_img != Color_fond, axis=-1)] = protec_img[np.any(protec_img != Color_fond, axis=-1)]

        min_size=50
        max_size=100
        all_size=[int(s) for s in np.arange(min_size,max_size,int((max_size-min_size)/21))]

        Illu_show = cv2.resize(Illu, [all_size[all_teams[Team]["size_efficiency"]]*3,all_size[all_teams[Team]["size_efficiency"]]*3], interpolation=cv2.INTER_NEAREST)
        Illu_show[np.any(Illu_show == [213, 158, 15], axis=-1)] = all_teams[Team]["col"]

        decal=max(2,int((Size_show-Illu_show.shape[0])/2))
        Illu_show=cv2.rotate(Illu_show,cv2.ROTATE_90_COUNTERCLOCKWISE)

        Illu_to_show[decal+Espacio_texto:Espacio_texto+decal+Illu_show.shape[0],Size_show*Team_count+decal:(Size_show*Team_count+decal+Illu_show.shape[0])]=Illu_show[:]

        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (0, 0, 0)
        thickness = 2

        text_width, text_height = cv2.getTextSize(Team, fontFace, fontScale, thickness)[0]
        CenterCoordinates = (int(Illu_show.shape[0] / 2) - int(text_width / 2))
        cv2.putText(Illu_to_show,Team,[Size_show*Team_count+decal+CenterCoordinates,25], # bottom left corner of text
        cv2.FONT_HERSHEY_SIMPLEX, # font to use
        1, # font scale
        (0, 0, 0), # color
        2, # line thickness
        )
        Illu = cv2.resize(Illu, [all_size[all_teams[Team]["size_efficiency"]], all_size[all_teams[Team]["size_efficiency"]]], interpolation=cv2.INTER_NEAREST)



        for h in herbivores[Team][:]:  # iterate over a copy
            h.Illu=Illu

        Team_count+=1


    cv2.imshow("Equipos", resize_img(Illu_to_show))
    cv2.moveWindow("Equipos", 0, 0)
    cv2.waitKey()

    try:
        cv2.destroyWindow("Equipos")
    except:
        pass



    #For visual representation
    brown = np.array([0, 50, 90], dtype=np.uint8)   # dark brown
    green = np.array([25, 152, 0], dtype=np.uint8)     # bright green


    result_video = cv2.VideoWriter("results.avi", cv2.VideoWriter_fourcc(*'XVID'), 1/dt, [int(width * block_size), int(height * block_size)])

    for t in np.arange(0,duration,dt):
        debut = time.time()
        #Environment

        Environment[1:height-1,1:width-1] = Environment[1:height-1,1:width-1] + (growth_rate)*255*dt
        Environment[Environment>255]=255


        cur_img = Environment.copy()
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR)
        cur_img = cur_img.astype(np.float32) / 255.0
        cur_img = (brown * (1 - cur_img) + green * cur_img).astype(np.uint8)
        cur_img = cv2.resize(cur_img, [int(width * block_size), int(height * block_size)], interpolation=cv2.INTER_NEAREST)

        if t%5==0:
            print("Time: " + str(round((t/duration)*100,2)))
            for Team in herbivores.keys():
                print("Team: "+Team+" has "+str(len(herbivores[Team]))+" individuals alive")
            print(death_type)
            print("")



        #Animals
        nb_total=0
        for Team in herbivores.keys():
            new_herbivores=[]
            for h in herbivores[Team][:]:  # iterate over a copy
                nb_total+=1
                child=h.move_in_time(dt, Environment)
                if not child is None:
                    new_herbivores.append(child)
                    total_nb_data[Team]+=1
                if h.alive<0:
                    if h.death_reason in death_type[Team]:
                        death_type[Team][h.death_reason]+=1
                    else:
                        death_type[Team][h.death_reason]=1

                    total_feeding[Team]+=h.amount_eaten
                    if h.age>(h.maturation*h.lifespan):#If the animal was adult, did he reproduce
                        total_repro[Team].append(h.nb_repro)

                    herbivores[Team].remove(h)
                else:
                    cx = int(h.pos_X * block_size + block_size / 2)
                    cy = int(h.pos_Y * block_size + block_size / 2)

                    illu_rot = rotate_image(h.Illu.copy(), - (h.angle*180)/math.pi)
                    illuh,illuw,_=illu_rot.shape
                    new_x=int(cx-illuw/2)
                    new_y = int(cy - illuw / 2)

                    repreh,reprew,_=cur_img.shape

                    illuX0=0
                    if new_x<0:
                        illuw=illuw+new_x
                        illuX0=-new_x
                        new_x=0
                    elif new_x+illuw>=reprew:
                        illuw-=(new_x+illuw)-reprew


                    illuY0=0
                    if new_y<0:
                        illuh=illuh+new_y
                        illuY0=-new_y
                        new_y=0

                    elif new_y+illuh>=repreh:
                        illuh-=(new_y+illuh)-repreh


                    illu_rot[np.any(illu_rot == [213,158,15], axis=-1)]=h.col
                    mask = np.any(illu_rot != Color_fond, axis=-1)

                    cur_img[new_y:new_y+illuh,new_x:illuw+new_x][mask[illuY0:illuY0+illuh,illuX0:illuX0+illuw]]=illu_rot[illuY0:illuY0+illuh,illuX0:illuX0+illuw][mask[illuY0:illuY0+illuh,illuX0:illuX0+illuw]]
            herbivores[Team].extend(new_herbivores)
        cv2.imshow("Survis live!", resize_img(cur_img))
        cv2.moveWindow("Survis live!", 0,0)
        to_wait=max(1,int(1000*(dt-(time.time()-debut))))
        cv2.waitKey(to_wait)

        if nb_total<1 or keyboard.is_pressed("q"):
            cv2.destroyWindow("Survis live!")
            break
        #result_video.write(cur_img)



    Text_Area=np.zeros([600,Size_show*len(all_teams.keys()),3], dtype=np.uint8)
    Text_Area[:]=Color_fond

        #We display the different kind of success:

    #Survivors teams:
    New_text=Text_Area.copy()
    nb_alive=[]
    Team_count = 0

    for Team in all_teams.keys():
        nb_alive.append(len(herbivores[Team]))
        if round_G==0:
            All_nb_alive.append(len(herbivores[Team]))
        else:
            All_nb_alive[Team_count]+=len(herbivores[Team])
        Text1 = "Survivors: " + str(len(herbivores[Team]))
        New_text=cv2.putText(New_text,Text1,[Size_show*Team_count+35,50], # bottom left corner of text
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        #Pie chart:
        Nb_death=sum(death_type[Team].values())
        values=[]
        colors=[]
        labels=[]
        for death in death_type[Team].keys():
            prop_deaths=death_type[Team][death]/Nb_death
            values.append(death_type[Team][death])
            if death=="Old age":
                colors.append((255,255,255))
            elif death=="Not enought food":
                colors.append((50,150,50))
            else:
                for tkill in all_teams.keys():
                    if death=="Eaten by "+ tkill+"'s":
                        colors.append(all_teams[tkill]["col"])
            labels.append(death)

        Pie_Illu=Do_pie.do_pie(values,colors,labels,Color_fond, Size_show)

        New_text[75:200+75,Size_show*Team_count:Size_show*(Team_count+1)]=Pie_Illu
        Team_count += 1

    Legend = cv2.imread(resource_path(os.path.join("Images", "Legen_death.png")))
    Legend=cv2.resize(Legend,[400,200])
    New_text[300:500,int((Size_show * len(all_teams.keys()))/2) - 200:int((Size_show * len(all_teams.keys()))/2) + 200]=Legend

    New_text = cv2.putText(New_text, "Number of Survis remaining", [int((Size_show * len(all_teams.keys()))/2) - 250, 525],  # bottom left corner of text
                           cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 4)

    New_Illu=Illu_to_show.copy()
    Team_count = 0
    rank=len(nb_alive) - rankdata(nb_alive).astype(int) +1
    for Team in all_teams.keys():
        if rank[Team_count]<4:
            Couronne = cv2.imread(resource_path(os.path.join("Images", "Couronne_"+str(rank[Team_count])+".png")))
            Couronne=cv2.resize(Couronne,[102,40])
            New_Illu[50:90,Size_show*Team_count+int(Size_show/2)-51:Size_show*Team_count+int(Size_show/2)+51]=Couronne
        Team_count+=1

    New_Illu=np.vstack([New_Illu,New_text])
    cv2.imshow("Results survival at the end", resize_img(New_Illu))
    cv2.moveWindow("Results survival at the end", 0, 0)
    cv2.waitKey()
    try:
        cv2.destroyWindow("Results survival at the end")
    except:
        pass


    #Number of Survis who ever lived
    New_text = Text_Area.copy()
    Nb_ever_living = []
    Team_count = 0

    for Team in all_teams.keys():
        Nb_ever_living.append(total_nb_data[Team])
        if round_G==0:
            All_Nb_ever_living.append(total_nb_data[Team])
        else:
            All_Nb_ever_living[Team_count]+=total_nb_data[Team]

        Text1 = "Total number: " + str(total_nb_data[Team])
        New_text = cv2.putText(New_text, Text1, [Size_show * Team_count + 35, 50],  # bottom left corner of text
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        Team_count += 1

    New_Illu=Illu_to_show.copy()
    Team_count = 0
    rank=len(Nb_ever_living) - rankdata(Nb_ever_living).astype(int) +1
    for Team in all_teams.keys():
        if rank[Team_count]<4:
            Couronne = cv2.imread(resource_path(os.path.join("Images", "Couronne_"+str(rank[Team_count])+".png")))
            Couronne=cv2.resize(Couronne,[102,40])
            New_Illu[50:90,Size_show*Team_count+int(Size_show/2)-51:Size_show*Team_count+int(Size_show/2)+51]=Couronne
        Team_count+=1

    New_text = cv2.putText(New_text, "Number of Survis who ever lived", [int((Size_show * len(all_teams.keys()))/2) - 250, 250],  # bottom left corner of text
                           cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 4)
    
    New_Illu=np.vstack([New_Illu,New_text])
    cv2.imshow("Results number of Survis who lived", resize_img(New_Illu))
    cv2.moveWindow("Results number of Survis who lived", 0, 0)
    cv2.waitKey()

    try:
        cv2.destroyWindow("Results number of Survis who lived")
    except:
        pass

    # Best reproductor
    New_text = Text_Area.copy()
    Nb_babies = []
    Team_count = 0
    for Team in all_teams.keys():
        for Survi in herbivores[Team]:
            if Survi.alive:
                total_feeding[Team]+=Survi.amount_eaten
                total_repro[Team].append(Survi.nb_repro)

    for Team in all_teams.keys():
        Nb_babies.append(np.mean(total_repro[Team]))
        if round_G == 0:
            All_offspring.append(total_repro[Team])
        else:
            All_offspring[Team_count]=All_offspring[Team_count]+total_repro[Team]

        Text1 = "Offspring: " + str(round(np.mean(total_repro[Team]),2))
        New_text = cv2.putText(New_text, Text1, [Size_show * Team_count + 35, 50],  # bottom left corner of text
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        Team_count += 1

    New_Illu = Illu_to_show.copy()
    Team_count = 0
    rank = len(Nb_babies) - rankdata(Nb_babies).astype(int) + 1
    for Team in all_teams.keys():
        if rank[Team_count] < 4:
            Couronne = cv2.imread(resource_path(os.path.join("Images", "Couronne_" + str(rank[Team_count]) + ".png")))
            Couronne = cv2.resize(Couronne, [102, 40])
            New_Illu[50:90, Size_show * Team_count + int(Size_show / 2) - 51:Size_show * Team_count + int(
                Size_show / 2) + 51] = Couronne
        Team_count += 1

    New_text = cv2.putText(New_text, "Number of offspring per adult",
                           [int((Size_show * len(all_teams.keys())) / 2) - 250, 250],  # bottom left corner of text
                           cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 4)

    New_Illu = np.vstack([New_Illu, New_text])
    cv2.imshow("Offspring", resize_img(New_Illu))
    cv2.moveWindow("Offspring", 0, 0)
    cv2.waitKey()

    try:
        cv2.destroyWindow("Offspring")
    except:
        pass




    #Top herbivore
    New_text = Text_Area.copy()


    Team_count = 0
    for Team in all_teams.keys():
        if round_G == 0:
            All_All_food_eaten.append(total_feeding[Team])
        else:
            All_All_food_eaten[Team_count] += total_feeding[Team]
        Team_count +=1

    print(All_All_food_eaten)


    All_food_eaten = []
    Team_count = 0
    for Team in all_teams.keys():
        if all_teams[Team]["type"]=="H":
            All_food_eaten.append(total_feeding[Team])
            Text1 = "Grass eaten: " + str(round(total_feeding[Team],2))
            New_text = cv2.putText(New_text, Text1, [Size_show * Team_count + 35, 50],  # bottom left corner of text
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        Team_count += 1

    New_Illu=Illu_to_show.copy()
    Team_count = 0
    rank=len(All_food_eaten) - rankdata(All_food_eaten).astype(int) +1
    print(rank)
    H_count=0
    for Team in all_teams.keys():
        if all_teams[Team]["type"] == "H":
            print("Herbi")
            if rank[H_count]<4:
                print("add couronne")
                Couronne = cv2.imread(
                    resource_path(os.path.join("Images", "Couronne_" + str(rank[H_count]) + ".png")))
                Couronne = cv2.resize(Couronne, [102, 40])
                New_Illu[50:90, Size_show * Team_count + int(Size_show / 2) - 51:Size_show * Team_count + int(
                    Size_show / 2) + 51] = Couronne
            H_count+=1

        Team_count+=1

    New_text = cv2.putText(New_text, "Best herbivorous", [int((Size_show * len(all_teams.keys()))/2) - 250, 250],  # bottom left corner of text
                           cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 4)

    New_Illu=np.vstack([New_Illu,New_text])
    cv2.imshow("Results of top herbivorous", resize_img(New_Illu))
    cv2.moveWindow("Results of top herbivorous", 0, 0)
    cv2.waitKey()

    try:
        cv2.destroyWindow("Results of top herbivorous")
    except:
        pass




    # Top Predator
    New_text = Text_Area.copy()
    All_food_eaten = []
    Team_count = 0
    for Team in all_teams.keys():
        if all_teams[Team]["type"] == "C":
            All_food_eaten.append(total_feeding[Team])
            Text1 = "Prey catched: " + str(round(total_feeding[Team], 2))
            New_text = cv2.putText(New_text, Text1, [Size_show * Team_count + 35, 50],  # bottom left corner of text
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        Team_count += 1

    New_Illu = Illu_to_show.copy()
    Team_count = 0
    rank = len(All_food_eaten) - rankdata(All_food_eaten).astype(int) + 1
    Pcount=0
    for Team in all_teams.keys():
        if all_teams[Team]["type"] == "C":
            if rank[Pcount] < 4:
                Couronne = cv2.imread(
                    resource_path(os.path.join("Images", "Couronne_" + str(rank[Pcount]) + ".png")))
                Couronne = cv2.resize(Couronne, [102, 40])
                New_Illu[50:90, Size_show * Team_count + int(Size_show / 2) - 51:Size_show * Team_count + int(
                    Size_show / 2) + 51] = Couronne
            Pcount+=1
        Team_count += 1

    New_text = cv2.putText(New_text, "Best carnivorous", [int((Size_show * len(all_teams.keys()))/2) - 250, 250],  # bottom left corner of text
                           cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 4)

    New_Illu = np.vstack([New_Illu, New_text])
    cv2.imshow("Results of Best carnivorous", resize_img(New_Illu))
    cv2.moveWindow("Results of Best carnivorous", 0, 0)
    cv2.waitKey()

    try:
        cv2.destroyWindow("Results of Best carnivorous")
    except:
        pass

    All_illus.append(Illu_to_show.copy())


for turn_img in All_illus:
    cv2.imshow("Changes through time", resize_img(turn_img))
    cv2.moveWindow("Changes through time", 0, 0)
    cv2.waitKey(1500)


Final_title= np.zeros([100, Size_show * len(all_teams.keys()), 3], dtype=np.uint8)
Final_title[:] = Color_fond
Final_title=cv2.putText(Final_title,"Final summary", [int((Size_show * len(all_teams.keys())) / 2) - 150, 40],
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)



def show_final_summary(Legend, values,title, who):
    New_text = Text_Area.copy()
    Team_count = 0
    internal_count=0
    sub_values=[]
    for Team in all_teams.keys():
        print(all_teams[Team]["type"] in who)
        print(who)
        print(all_teams[Team]["type"])
        if all_teams[Team]["type"] in who:
            Text1 =  Legend + str(values[Team_count])
            New_text = cv2.putText(New_text, Text1, [Size_show * Team_count + 35, 50],  # bottom left corner of text
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            sub_values.append(values[Team_count])
            internal_count+=1
        Team_count += 1

    New_text = cv2.putText(New_text, title,
                           [int((Size_show * len(all_teams.keys())) / 2) - 250, 250],  # bottom left corner of text
                           cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 4)

    New_Illu = Illu_to_show.copy()
    Team_count = 0
    rank = len(sub_values) - rankdata(sub_values).astype(int) + 1
    internal_count=0
    for Team in all_teams.keys():
        if all_teams[Team]["type"] in who:
            if rank[internal_count] < 4:
                Couronne = cv2.imread(resource_path(os.path.join("Images", "Couronne_" + str(rank[internal_count]) + ".png")))
                Couronne = cv2.resize(Couronne, [102, 40])
                New_Illu[50:90, Size_show * Team_count + int(Size_show / 2) - 51:Size_show * Team_count + int(
                    Size_show / 2) + 51] = Couronne
                internal_count+=1
        Team_count += 1

    New_Illu = np.vstack([Final_title,New_Illu, New_text])
    cv2.imshow(title, resize_img(New_Illu))
    cv2.moveWindow(title, 0, 0)
    cv2.waitKey()

    try:
        cv2.destroyWindow("Results survival at the end")
    except:
        pass

show_final_summary(Legend="Survivors: ", values=All_nb_alive,title="Number of Survis remaining", who=["C","H"])
show_final_summary(Legend="Total number: ", values=All_Nb_ever_living,title="Number of Survis who ever lived", who=["C","H"])

#Offspring all averages:
values_offspring=[round(np.mean(values),2) for values in All_offspring]
show_final_summary(Legend="Offspring: ", values=values_offspring,title="Offspring", who=["C","H"])

show_final_summary(Legend="Best herbivorous: ", values=All_All_food_eaten,title="Best herbivorous", who=["H"])
show_final_summary(Legend="Best carnivorous: ", values=All_All_food_eaten,title="Best carnivorous", who=["C"])
