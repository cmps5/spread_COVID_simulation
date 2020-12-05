import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import math
from matplotlib.animation import FuncAnimation
import datetime
import csv
import matplotlib.animation as animation
from postgis import Polygon,MultiPolygon
from postgis.psycopg import register
import matplotlib.gridspec as gridspec

#memory profiler
import resource
def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                usage[2]/1024.0 )


#set global arrays that hold the border limits for the dynamic zoom
og_border = [-120000, 165000, -310000, 285000]
curr_border = [-120000, 165000, -310000, 285000]
new_border = []
quant = []

#function that updates the "curr_border" structure with the right coordinates
def get_new_border(i):
    global curr_border
    global og_border
    global new_border
    global quant

    #Define the frequency of the zoom and the smoothness of it
    periodo = 720
    t = 150

    if i%periodo == 0:
        #get the district with the highest number of infections
        index = count_infected[i].index(max(count_infected[i]))

        x_min = distrito_coords[index][0]
        y_min = distrito_coords[index][1]
        x_max = distrito_coords[index][2]
        y_max = distrito_coords[index][3]

        #set the "goal" border to that district's border
        new_border = [x_min,x_max,y_min,y_max]

        #set an array with the values needed to add to each coordinate in order to achieve the district zoom
        temp = [abs((-120000 - x_min)/t),abs((165000 - x_max)/t),abs((-310000 - y_min)/t),abs((285000 - y_max)/t)]
        quant = temp[:]

    x_min,x_max,y_min,y_max = curr_border[0], curr_border[1], curr_border[2], curr_border[3]

    #Zoomin
    if i> periodo and i%periodo <= t:
        new_x_min,new_x_max,new_y_min,new_y_max = new_border[0], new_border[1], new_border[2], new_border[3]

        if x_min < new_x_min:
            x_min = x_min + quant[0]
        if x_max > new_x_max:
            x_max = x_max - quant[1]
        if y_min < new_y_min:
            y_min = y_min + quant[2]
        if y_max > new_y_max:
            y_max = y_max - quant[3]

    else:
        new_x_min,new_x_max,new_y_min,new_y_max = og_border[0], og_border[1], og_border[2], og_border[3]
        #Zoomout
        if i%periodo > periodo/2:
            if x_min != new_x_min or x_max != new_x_max or y_min != new_y_min or y_max != new_y_max:
                if x_min > new_x_min:
                    x_min = x_min - quant[0]
                if x_max < new_x_max:
                    x_max = x_max + quant[1]
                if y_min > new_y_min:
                    y_min = y_min - quant[2]
                if y_max < new_y_max:
                    y_max = y_max + quant[3]

    #update border to be used at timestamp i
    curr_border = [x_min,x_max,y_min,y_max]


#function that animates the plots based on a timestamp i
def animate(i):
	print(i)
	#set the map's title to the timestamp's date form
	ax.set_title(datetime.datetime.utcfromtimestamp(ts_i+i*10))

	#get the new border for the map
	get_new_border(i)
	ax.set(xlim=(curr_border[0],curr_border[1]),ylim=(curr_border[2], curr_border[3]))

	#create a color array for the taxis based on their state at timestamp i
	tracks = []
	color_array = []
	for j in range(len(states[i])):
		if states[i][j] == 1:
			tracks.append(offsets[i][j])
			color_array.append((0.92,0,0))
		if states[i][j] == 0:
			color_array.append((0.20,0.66,0.28))

	for j in range(len(offsets[0])):
		if offsets[i][j][0] == 0 and offsets[i][j][1] == 0:
			color_array[j] = (1,1,1)

	#update the taxis position based on the offsets for timestamp i and update their color
	scat.set_offsets(offsets[i])
	scat.set_color(np.array(color_array))


	#set the case evolution chart's x and y ticks and labels acording to the current time and the max value of cases
	ax2.set_xticks([0,i])
	temp = [0,i]
	labels = []

	for label in temp:
		labels.append(datetime.datetime.utcfromtimestamp(ts_i+label*10).strftime("%H:%M"))

	ax2.set_xticklabels(labels)
	ax2.set_yticks([0,max(points_contaminados[i][1])+1])
	#set new data to the case evolution chart based on the timestamp i
	line.set_data(points_contaminados[i][0],points_contaminados[i][1])




	ax6.set_xticks([0,i])
	temp = [0,i]
	labels = []

	for label in temp:
		labels.append(datetime.datetime.utcfromtimestamp(ts_i+label*10).strftime("%H:%M"))

	ax6.set_xticklabels(labels)
	ax6.set_yticks([0,max(r0[i][1])+1])
	#set new data to the case evolution chart based on the timestamp i
	line2.set_data(r0[i][0],r0[i][1])




	ax3.clear()
	ax3.barh(y_ch, main_infected[i], align='center')
	ax3.set_yticks(y_ch)
	ax3.set_yticklabels(main_dist)
	ax3.invert_yaxis()  # labels read top-to-bottom
	ax3.set_title('Infeções por Distrito')


	#get offsets for every infected taxi at timestamp i
	xs,ys = [],[]
	for coords in tracks:
		xs.append(coords[0])
		ys.append(coords[1])

	#plot those coordinates in each of the scatter "roadmaps" for Porto and Lisboa
	ax4.scatter(xs,ys,s=1,color='red')
	ax5.scatter(xs,ys,s=1,color='red')


#define the sampling "step" to be used
step = 10
conn = psycopg2.connect("dbname=cmps user=cmps")
register(conn)
cursor_psql = conn.cursor()

#query the database for all existing taxis in the tracks table and store their indexes/keys
sql = """select distinct taxi from tracks order by 1"""
cursor_psql.execute(sql)
results = cursor_psql.fetchall()
taxis_x ={}

array_size = int(24*60*60/step)

for row in results:
	taxis_x[int(row[0])] = np.zeros(array_size)

taxis_l=list(taxis_x.keys())

#load the offsets array into memory from a file
offsets = []
with open('offsets.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    i = 0
    for row in reader:
        l = []
        for j in row:
            x,y = j.split()
            x = float(x)
            y = float(y)
            l.append([x,y])
        offsets.append(l)

#offsets=offsets.copy()[:2000]
#load the interactions array into memory from a file
interactions = np.load('interactions.npy', allow_pickle=True)

#query the cont_aad_caop2018 table to find each district's envelope and polygon
cursor_psql = conn.cursor()
sql = "select distrito,st_envelope(st_union(proj_boundary)),st_astext(st_union(proj_boundary)) from cont_aad_caop2018 group by distrito"

cursor_psql.execute(sql)
results = cursor_psql.fetchall()

#create data structures to hold the names, coordinates, polygons and number of infections per time, accordingly
distrito_nome = []
distrito_coords = []
distrito_sql = []
count_infected = []

#for every district, set their infections at 0 at all times
for i in range(len(offsets)):
	count_infected.append([])
	for j in range(len(results)):
		count_infected[i].append(0)

#process the results obtained by the query
for row in results:
	envelope = row[1]
	xys = envelope.coords[0]
	xs,ys = [],[]
	for (x,y) in xys:
		xs.append(x)
		ys.append(y)

	#from the envelope, obtain the min and max of each axis
	x_max = max(xs)
	x_min = min(xs)
	y_max = max(ys)
	y_min = min(ys)

	#manipulate the values in order to obtain values acording to the scale of the map
	comprimento = abs(x_max - x_min)
	altura = abs(y_max - y_min)
	addon = (comprimento * 2.09 - altura)/2
	y_max = y_max + addon
	y_min = y_min - addon

	#append the values to the respective lists
	distrito_nome.append(row[0])
	distrito_coords.append((x_min,y_min,x_max,y_max))
	distrito_sql.append(row[2])

# first 2 infections at random
conn = psycopg2.connect("dbname=cmps user=cmps")
register(conn)
cursor_psql = conn.cursor()

#select first 10 taxis in PORTO
sql3 = """select  taxi , min(ts) from tracks where st_contains((select st_collect(proj_boundary) from cont_aad_caop2018 where concelho='PORTO'), st_startpoint(proj_track)) group by taxi  ORDER BY min(ts)  limit 40;
"""

cursor_psql.execute(sql3)
results_i = cursor_psql.fetchall()
t_p=[]
for taxi in results_i:
	t_p.append(taxi[0])

inf_taxi=np.zeros(len(t_p))

for t in range(1000):
    for ta in range(len(interactions[0])):
        if len(interactions[t][ta]) != 0:
            for taxi in t_p:
                if taxis_l.index(int(taxi)) == ta:
                    inf_taxi[t_p.index(str(taxi))] += len(interactions[t][ta])
                elif taxi in interactions[t][ta]:
                    inf_taxi[t_p.index(str(taxi))] += 1

t_P=[]
for i in range(10):

    M = int(np.where(inf_taxi == inf_taxi.max())[0][0])
    t_P.append(t_p[M])
    inf_taxi=np.delete(inf_taxi,M)
    t_p.pop(M)


print(t_P)


#select first 10 taxis in LISBON
sql4 = """select  taxi , min(ts) from tracks where st_contains((select st_collect(proj_boundary) from cont_aad_caop2018 where concelho='LISBOA'), st_startpoint(proj_track)) group by taxi  ORDER BY min(ts)  limit 40;
"""
cursor_psql.execute(sql4)
results_j = cursor_psql.fetchall()
t_l=[]
for taxi in results_j:
	t_l.append(taxi[0])

inf_taxi2=np.zeros(len(t_l))

for t in range(1000):
    for ta in range(len(interactions[0])):
        if len(interactions[t][ta]) != 0:
            for taxi in t_l:
                if taxis_l.index(int(taxi)) == ta:
                    inf_taxi2[t_l.index(str(taxi))] += len(interactions[t][ta])
                elif taxi in interactions[t][ta]:
                    inf_taxi2[t_l.index(str(taxi))] += 1

t_L=[]
for i in range(10):

    M = int(np.where(inf_taxi2 == inf_taxi2.max())[0][0])
    print(t_l[M])
    t_L.append(t_l[M])
    inf_taxi2=np.delete(inf_taxi2,M)
    t_l.pop(M)







print(t_L)

# select 2 at random, one from each district
import random as rd
rd.seed(28)
t_P , t_L = ['20000239'] , ['20092034']
start_infection = []
start_infection.append(int(rd.choice(t_P)))
start_infection.append(int(rd.choice(t_L)))
#update the number of infections at Porto and Lisboa districts by 1 for all timestamps
for i in range(len(offsets)):
	count_infected[i][distrito_nome.index('PORTO')] = 1
	count_infected[i][distrito_nome.index('LISBOA')] = 1
print(start_infection)

#find index in array
i_infection =[0,0]
i_infection[0] , i_infection[1] = taxis_l.index(start_infection[0]) , taxis_l.index(start_infection[1])
print(i_infection)


#create data structure to hold the state of each taxi acording to a timestamp i, set the probability of propagation and create a data structure to
#hold the total number of infected taxis over time
states=[]
prob = 0.02
n_contaminados = []

for i in range(len(offsets)):
	n_contaminados.append(2)

for time in range(len(offsets)):
	states.append([])
	for taxi in range(len(offsets[0])):
		if taxi == i_infection[0] or taxi == i_infection[1]:
			states[time].append(1)
		else:
			states[time].append(0)

print(len(states),len(offsets),len(offsets[0]),len(states[0]),len(interactions),len(interactions[0]))


#Infection simulation
#for ts in range(len(interactions)):
for ts in range(len(offsets)):
	for index in range(len(interactions[0])):
		#for every taxi that interacts at timestamp i
		for taxi in interactions[ts][index]:
			#if taxi a is infected and interacts with a non infected taxi b
			if states[ts][index] == 1 and states[ts][taxi] == 0:
				if rd.random() <= prob:

					#increase the number of total infections
					for i in range(ts+1,len(offsets)):
						n_contaminados[i] = n_contaminados[i] + 1

					#update b's future states
					for l in range(ts+1,len(offsets)):
						states[l][taxi] = 1

					#determine in what district occurred the infection
					for j in range(len(distrito_coords)):
						sql = """select st_within(ST_GeomFromText('POINT(""" + str(offsets[ts][taxi][0])+ " " + str(offsets[ts][taxi][1]) + """)',3763), ST_GeomFromText('""" + distrito_sql[j] + """',3763));"""
						cursor_psql.execute(sql)
						results = cursor_psql.fetchall()

						if results[0][0] == True:
							#increase the number of infections in that district
							for k in range(ts+1,len(offsets)):
								count_infected[k][j] = count_infected[k][j] + 1
							break

			#if taxi a in not infected and interacts with an infected taxi a
			if states[ts][index] == 0 and states[ts][taxi] == 1:
				if rd.random() <= prob:

					#increase the number of total infections
					for i in range(ts+1,len(offsets)):
						n_contaminados[i] = n_contaminados[i] + 1

					#update a's future states
					for l in range(ts+1,len(offsets)):
						states[l][index] = 1

					#determine in what district occurred the infection
					for j in range(len(distrito_coords)):
						sql = """select st_within(ST_GeomFromText('POINT(""" + str(offsets[ts][index][0])+ " " + str(offsets[ts][index][1]) + """)',3763), ST_GeomFromText('""" + distrito_sql[j] + """',3763));"""
						cursor_psql.execute(sql)
						results = cursor_psql.fetchall()

						if results[0][0] == True:
							#increase the number of infections in that district
							for k in range(ts+1,len(offsets)):
								count_infected[k][j] = count_infected[k][j] + 1
							break


#define starting timestamp and scale
ts_i=1570665600
scale=1/3000000

xs_min, xs_max, ys_min, ys_max = -120000, 165000, -310000, 285000
width_in_inches = (xs_max-xs_min)/0.0254*1.1
height_in_inches = (ys_max-ys_min)/0.0254*1.1

# Create 2x3 sub plots
gs = gridspec.GridSpec(3, 3)

#set figure size
fig = plt.figure(figsize=(width_in_inches*3*scale + 4 , height_in_inches*scale))

ax = fig.add_subplot(gs[0:3, 0]) # span all rows, col 0

ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 2

ax3 = fig.add_subplot(gs[2, 1:3]) # row 1, col 2

ax4 = fig.add_subplot(gs[1, 1]) # row 0, col 1

ax5 = fig.add_subplot(gs[1, 2]) # row 1, col 1

ax6 = fig.add_subplot(gs[0, 2]) # row 0, col 2

#set map's axis visibility and limits
ax.axis('off')
ax.set(xlim=(xs_min, xs_max), ylim=(ys_min, ys_max))

#get every district's polygon/multipolygon from the cont_aad_caop2018 table
cursor_psql = conn.cursor()
sql = "select distrito,st_union(proj_boundary) from cont_aad_caop2018 group by distrito"

cursor_psql.execute(sql)
results = cursor_psql.fetchall()

#process the results, draw Portugal's map district by district
xs , ys = [],[]
for row in results:
    geom = row[1]
    if type(geom) is MultiPolygon:
        for pol in geom:
            xys = pol[0].coords
            xs, ys = [],[]
            for (x,y) in xys:
                xs.append(x)
                ys.append(y)
			#plot if is polygon
            ax.plot(xs,ys,color='black',lw='0.2')
    if type(geom) is Polygon:
        xys = geom[0].coords
        xs, ys = [],[]
        for (x,y) in xys:
            xs.append(x)
            ys.append(y)
		#plot if multipolygon
        ax.plot(xs,ys,color='black',lw='0.2')

#get timestamp 0 offsets and colors
x,y = [],[]
for i in offsets[0]:
	x.append(i[0])
	y.append(i[1])

color_array = []
for state in states[0]:
	if state == 1:
		color_array.append((0.92,0,0))
	if state == 0:
		color_array.append((0.20,0.66,0.28))

#plot the taxis at timestamp 0
scat = ax.scatter(x,y,s=2,color=color_array)

#create data structures to hold the number of infections per timestamp
contaminados = []
points_contaminados = []
r = []
r0=[]
#update the values of the data structures
for i in range(len(offsets)):
	contaminados.append((i,n_contaminados[i]))
	temp1 = []
	temp2 = []
	for j in range(0,i+1):
		temp1.append(contaminados[j][0])
		temp2.append(contaminados[j][1])
	points_contaminados.append([])
	points_contaminados[i].append(temp1)
	points_contaminados[i].append(temp2)
	if i >= 1:
		r_0=(n_contaminados[i]-n_contaminados[i-1])/n_contaminados[i-1]
		r.append((i,r_0))
		temp_1 = []
		temp_2 = []
		for j in range(0,i+1):
			temp_1.append(r[j][0])
			temp_2.append(r[j][1])
		r0.append([])
		r0[i].append(temp_1)
		r0[i].append(temp_2)
		if i==3000:
			print(n_contaminados[i],'contaminados')

	else:
		r.append((i,0))
		temp_1 = []
		temp_2 = []
		for j in range(0,i+1):
			temp_1.append(r[j][0])
			temp_2.append(r[j][1])
		r0.append([])
		r0[i].append(temp_1)
		r0[i].append(temp_2)



#plot the line of the number of infections at timestamp 0
line, = ax2.plot(0,n_contaminados[0],color = "red")

#set the subplot's title, xticks and yticks
ax2.set_title('Infeções ao longo do tempo')
ax2.set_xticks(np.arange(0, 1))
ax2.set_yticks(np.arange(0, 3))




#plot the r0 line  at timestamp 0
line2, = ax6.plot(0,0,color = "red")

#set the subplot's title, xticks and yticks
ax6.set_title('r0 ao longo do tempo')
ax6.set_xticks(np.arange(0, 1))
ax6.set_yticks(np.arange(0, 3))

# Infected chart by district
main_infected = []
main_dist = ('AVEIRO', 'BRAGA', 'COIMBRA', 'LEIRIA', 'LISBOA', 'PORTO', 'SANTARÉM', 'SETÚBAL')

for i in range(len(count_infected)):
	main_infected.append([])
	for j in range(len(main_dist)):
		main_infected[i].append(0)

for i in range(len(count_infected)):
	main_infected[i][0] = count_infected[i][distrito_nome.index('AVEIRO')]
	main_infected[i][1] = count_infected[i][distrito_nome.index('BRAGA')]
	main_infected[i][2] = count_infected[i][distrito_nome.index('COIMBRA')]
	main_infected[i][3] = count_infected[i][distrito_nome.index('LEIRIA')]
	main_infected[i][4] = count_infected[i][distrito_nome.index('LISBOA')]
	main_infected[i][5] = count_infected[i][distrito_nome.index('PORTO')]
	main_infected[i][6] = count_infected[i][distrito_nome.index('SANTARÉM')]
	main_infected[i][7] = count_infected[i][distrito_nome.index('SETÚBAL')]

y_ch = np.arange(len(main_dist))

ax3.barh(y_ch, main_infected[0], align='center')
ax3.set_yticks(y_ch)
ax3.set_yticklabels(main_dist)
ax3.invert_yaxis()  # labels read top-to-bottom
ax3.set_title('Infeções por Distrito')


#get the Porto concelho envelope and polygon from the cont_aad_caop2018 table
sql = "select st_union(proj_boundary),st_envelope(st_union(proj_boundary)) from cont_aad_caop2018 where concelho = 'PORTO'"
cursor_psql.execute(sql)
results = cursor_psql.fetchall()

#get the min and max on both axis using the evelope
envelope = results[0][1]
xys = envelope.coords[0]
xs,ys = [],[]
for (x,y) in xys:
	xs.append(x)
	ys.append(y)

x_max = max(xs)
x_min = min(xs)
y_max = max(ys)
y_min = min(ys)

#set subplot's limits to the min/max values obtained
ax4.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

#get the property lines to plot of the Porto concelho
geom = results[0][0]
xys = geom[0].coords
xs, ys = [],[]
for (x,y) in xys:
	xs.append(x)
	ys.append(y)

#plot said lines
ax4.plot(xs,ys,color='black',lw='0.5')

#format plot's title, xticks, yticks and labels
ax4.set_title('Concelho do Porto')
ax4.set_yticklabels([])
ax4.set_xticklabels([])
ax4.tick_params(length=0,labelbottom=False,labelleft=False,labelright=False,labeltop=False)


#get the offsets of the infected taxis at timestamp 0
tracks = []

for j in range(len(offsets[0])):
	if states[0][j] == 1:
		tracks.append(offsets[0][j])

xsx,ysx = [],[]
for coords in tracks:
	xsx.append(coords[0])
	ysx.append(coords[1])

#plot those offsets on top of Porto's bounderies
ax4.scatter(xsx,ysx,s=1,color='red')

#get the Lisboa concelho envelope and polygon from the cont_aad_caop2018 table
sql = "select st_union(proj_boundary),st_envelope(st_union(proj_boundary)) from cont_aad_caop2018 where concelho = 'LISBOA'"
cursor_psql.execute(sql)
results = cursor_psql.fetchall()

#get the min and max on both axis using the evelope
envelope = results[0][1]
xys = envelope.coords[0]
xs,ys = [],[]
for (x,y) in xys:
	xs.append(x)
	ys.append(y)

x_max = max(xs)
x_min = min(xs)
y_max = max(ys)
y_min = min(ys)

#set subplot's limits to the min/max values obtained
ax5.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

#get the property lines to plot of the Porto concelho
geom = results[0][0]
xys = geom[0].coords
xs, ys = [],[]
for (x,y) in xys:
	xs.append(x)
	ys.append(y)

#plot said lines
ax5.plot(xs,ys,color='black',lw='0.5')

#format plot's title, xticks, yticks and labels
ax5.set_title('Concelho de Lisboa')
ax5.set_yticklabels([])
ax5.set_xticklabels([])
ax5.tick_params(length=0,labelbottom=False,labelleft=False,labelright=False,labeltop=False)

#plot the previously obtained offsets of infected taxis on top of Lisboa's bounderies
ax5.scatter(xsx,ysx,s=1,color='red')

#set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#print usage
print(using("before animation"))

#set up the animation parameters
anim = FuncAnimation(
	fig, animate, interval=10, frames=len(offsets)-1, repeat = False)

#save resulting animation to file
anim.save('anim.mp4', writer=writer)

#start
plt.draw()
plt.show()
