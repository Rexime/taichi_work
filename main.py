from operator import inv
from scene import Scene  #光线相关
import taichi as ti
from taichi.math import * #数学库

scene = Scene(voxel_edges=0.05,exposure=1)
scene.set_floor(-1, (1.0, 1.0, 1.0))
# scene.set_directional_light((2, 1, 1), 0.1, (1, 1, 1))
# scene.set_background_color((0.3, 0.4, 0.6))

@ti.func
def drawHeart(n,x_,y_,z_,inverse=False):
    for i, j, k in ti.ndrange((-n, n), (-n, n), (0, n)):
        x=i/n*2
        z=j/n*2
        y=k/n*2
        if (x*x+9.0/4*y*y+z*z-1)**3 - x*x*z*z*z-9.0/80*y*y*z*z*z<=0:
            if not inverse:
                scene.set_voxel(vec3(i+x_, j+y_, k+z_), 2, vec3(1.0, 0.0, 0.0))
            else:
                scene.set_voxel(vec3(-(i+x_), -(j+y_), k+z_), 2, vec3(1.0, 0.0, 0.0))

@ti.func
def drawCard(n):
    for i, j, k in ti.ndrange((-n*2/3, n*2/3), (-n, n), (-1, 1)):
        mat,color=scene.get_voxel(vec3(i, j, k))
        if mat==0 and i*i+j*j*0.77 < (n+1)*(n+1):
            scene.set_voxel(vec3(i, j, k), 1, vec3(1.0, 1.0, 1.0))
    for i, j, in ti.ndrange((-n*2/3, n*2/3), (-n, n)):
        if i*i+j*j*0.77 < (n+1)*(n+1):
            if abs(i)<n*2/3-6 and abs(j)<n-6 and i*i+j*j*0.77 < (n-4)*(n-4) :
                if (i+j)%6==0 or (i+j)%6==1:
                    scene.set_voxel(vec3(i, j, -2), 1, vec3(1.0, 1.0, 0.0))
                if (i+j)%6==2 or (i+j)%6==3:
                    scene.set_voxel(vec3(i, j, -2), 1, vec3(1.0, 0.5, 0.0))
                if (i+j)%6==4 or (i+j)%6==5:
                    scene.set_voxel(vec3(i, j, -2), 1, vec3(1.0, 0.3, 0.0))
            else:
                scene.set_voxel(vec3(i, j, -2), 1, vec3(1.0, 1.0, 1.0))

@ti.func            
def drawA(n,pos=(0,0,0),inverse=False):
    for i,j,k in ti.ndrange((-n, n), (-1.5*n, 1.5*n), (0, 2)):
            if i<0 and abs(j-i*3-1.5*n)<=2:
                if not inverse:
                    scene.set_voxel(vec3(i+pos[0], j+pos[1], k+pos[2]), 2, vec3(1.0, 0.0, 0.0))
                else:
                    scene.set_voxel(vec3(-(i+pos[0]), -(j+pos[1]), k+pos[2]), 2, vec3(1.0, 0.0, 0.0))
            elif i>=0 and abs(j+i*3+3-1.5*n)<=2:
                if not inverse:
                    scene.set_voxel(vec3(i+pos[0], j+pos[1], k+pos[2]), 2, vec3(1.0, 0.0, 0.0))
                else:
                    scene.set_voxel(vec3(-(i+pos[0]), -(j+pos[1]), k+pos[2]), 2, vec3(1.0, 0.0, 0.0))
            elif j+2==0 and i>=-n/2 and i<=n/2:  
                if not inverse:
                    scene.set_voxel(vec3(i+pos[0], j+pos[1], k+pos[2]), 2, vec3(1.0, 0.0, 0.0))
                else:
                    scene.set_voxel(vec3(-(i+pos[0]), -(j+pos[1]), k+pos[2]), 2, vec3(1.0, 0.0, 0.0))
        
@ti.kernel
def initialize_voxels():
    drawA(4,(-30,50,0))
    drawA(4,(-30,50,0),True)
    drawHeart(9,-30,35,0)
    drawHeart(9,-30,35,0,True)
    drawHeart(22,0,0,0)
    drawCard(60)
    

initialize_voxels()

scene.finish()
