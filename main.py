import random

import cv2
import numpy as np
import Class_Herbivorous
import math
import time
import Prepare_dataset

Color_fond = [80,208,146]


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=Color_fond)
  return result



#Time is in sec:
dt=1/20
duration=90#In sec


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


for round_G in range(5):
    if round_G==0:
        all_teams = Prepare_dataset.prepare_dataset(points=points_per_round[round_G], previous_data=None)
    else:
        all_teams=Prepare_dataset.prepare_dataset(points=points_per_round[round_G],previous_data=all_teams)

    death_type={Team:{} for Team in all_teams.keys()}
    total_nb_data={Team:0 for Team in all_teams.keys()}

    herbivores = {}
    for Team in all_teams.keys():
        if all_teams[Team]["type"]=="H":
            nb=20
            total_nb_data[Team]+=20
        else:
            nb=10
            total_nb_data[Team] += 5

        herbivores[Team]=[Class_Herbivorous.Herbivorous(height,width, Team,**all_teams[Team], age=random.random()*3) for _ in range(nb)]


    Body=cv2.imread("Images/Body.png")

    Size_show=300
    Espacio_texto=50
    Illu_to_show=np.zeros([Size_show+Espacio_texto,Size_show*len(all_teams.keys()),3], dtype=np.uint8)
    Illu_to_show[:]=Color_fond
    Team_count=0
    for Team in all_teams.keys():
        #Legs
        if all_teams[Team]["speed"]>0:
            Illu = cv2.imread("Images/legs"+str(all_teams[Team]["speed"])+".png")
            Illu=Illu[1:2344, 1:2344]
        else:
            Illu=np.zeros([2343,2343,3], dtype=np.uint8)
            Illu[:]=Color_fond

        #Mandibles
        if all_teams[Team]["mandi"] > 0:
            mandi_img = cv2.imread("Images/mandibles"+str(all_teams[Team]["mandi"])+".png")
            mandi_img=mandi_img[1:2344, 1:2344]
            Illu[np.any(mandi_img != Color_fond, axis=-1)] = mandi_img[np.any(mandi_img != Color_fond, axis=-1)]

        body_img=Body[1:2344, 1:2344].copy()
        Illu[np.any(body_img != Color_fond, axis=-1)] = body_img[np.any(body_img != Color_fond, axis=-1)]

        if all_teams[Team]["eye"] > 0:
            eye_img = cv2.imread("Images/eye"+str(all_teams[Team]["eye"])+".png")
            eye_img = eye_img[1:2344, 1:2344]
            eye_img2 = cv2.flip(eye_img, 0)

            Illu[np.any(eye_img != Color_fond, axis=-1)] = eye_img[np.any(eye_img != Color_fond, axis=-1)]
            Illu[np.any(eye_img2 != Color_fond, axis=-1)] = eye_img2[np.any(eye_img2 != Color_fond, axis=-1)]

        if all_teams[Team]["wings"]>1:
            wing_img = cv2.imread("Images/wings"+str(all_teams[Team]["wings"]-1)+".png")
            wing_img = wing_img[1:2344, 1:2344]
            Illu[np.any(wing_img != Color_fond, axis=-1)] = wing_img[np.any(wing_img != Color_fond, axis=-1)]

        if all_teams[Team]["protect"] > 0:
            protec_img = cv2.imread("Images/protection"+str(all_teams[Team]["protect"])+".png")
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
        cv2.putText(Illu_to_show,Team,[Size_show*Team_count+decal+CenterCoordinates,50], # bottom left corner of text
        cv2.FONT_HERSHEY_SIMPLEX, # font to use
        1, # font scale
        (0, 0, 0), # color
        2, # line thickness
        )
        Illu = cv2.resize(Illu, [all_size[all_teams[Team]["size_efficiency"]], all_size[all_teams[Team]["size_efficiency"]]], interpolation=cv2.INTER_NEAREST)



        for h in herbivores[Team][:]:  # iterate over a copy
            h.Illu=Illu

        Team_count+=1


    cv2.imshow("Equipos", Illu_to_show)
    cv2.waitKey()



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
        cv2.imshow("W", cur_img)
        to_wait=max(1,int(1000*(dt-(time.time()-debut))))
        cv2.waitKey(to_wait)

        if nb_total<1:
            break
        #result_video.write(cur_img)




    Text_Area=np.zeros([600,Size_show*len(all_teams.keys()),3], dtype=np.uint8)
    Text_Area[:]=Color_fond

    Team_count=0
    for Team in all_teams.keys():
        Nb_alive = len(herbivores[Team])
        Text1="Survivors: "+ str(Nb_alive)
        Text_Area=cv2.putText(Text_Area,Text1,[Size_show*Team_count+35,50], # bottom left corner of text
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        Nb_ever_living=total_nb_data[Team]
        Text2="Nb who lived: "+ str(Nb_ever_living)
        Text_Area=cv2.putText(Text_Area,Text2,[Size_show*Team_count+35,100], # bottom left corner of text
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


        Nb_death=sum(death_type[Team].values())
        shift=50
        for death in death_type[Team].keys():
            prop_deaths=death_type[Team][death]/Nb_death
            Text3 = death +":"
            Text4= str(round(prop_deaths*100,1))+"%"

            Text_Area = cv2.putText(Text_Area, Text3, [Size_show * Team_count + 35, 100+shift],  # bottom left corner of text
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            shift+=40
            Text_Area = cv2.putText(Text_Area, Text4, [Size_show * Team_count + 55, 100+shift],  # bottom left corner of text
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            shift+=50

        Team_count+=1

    Illu_to_show=np.vstack([Illu_to_show,Text_Area])
    cv2.imshow("Results", Illu_to_show)
    cv2.waitKey()



