# NeuralDynaRender
> 맞춤형 3D 시각화를 위한 NeRF, 볼륨 렌더링, 및 PBD 기술 융합과 데이터 출력 및 변형


## 설명

Neural Dyna Render는 실시간 볼륨 렌더링을 위한 도구이다. 이 프로젝트는 Neural Radiance Fields 기술을 활용해 2D 데이터를 3D 볼륨으로 변환, Parallel Resampling과 Raycasting 기술을 활용하여 하고, Position Based Dynamics(PBD) 물리 엔진을 통해 볼륨 데이터를 변형한다.

### 주요 기능
- 렌더링

기존의 ngp(neural graphics primitives)를 활용한 NeRF(Neural radiance fields)에서 제공하는 렌더링이 아닌, 자체적인 렌더링 모델을 제작하여 3D로 변환된 데이터를 출력한다. 사용자는 4가지 옵션 중에 하나를 선택하여 해당 옵션이 적용된 변형을 확인할 수 있다.


<table>
    <tbody>
    	<tr>
        	<th style="text-align: center">Normal</th>
            <th style="text-align: center"><img src="https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/b408bd79-bae1-4046-a8bb-ddb2a8979234" /></th>
        </tr>
		<tr>
        	<th style="text-align: center">Wave</th>
            <th style="text-align: center"><img src="https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/5cd10c07-9aaa-49b5-8b67-42207be6d103" /></th>
        </tr>
		<tr>
        	<th style="text-align: center">Twist</th>
            <th style="text-align: center"><img src="https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/a470f93f-4c63-4030-bab9-ba217670980a" /></th>
        </tr>
		<tr>
        	<th style="text-align: center">Bubble</th>
            <th style="text-align: center"><img src="https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/73ee0f3c-7a17-49be-bb47-30413ac19d99" /></th>
        </tr>
    </tbody>
</table>


- PBD

PBD란  position based dynamics의 약자로 위치 기반 물리엔진를 의미한다. 3D 데이터는 2D 데이터보다 좀 더 입체감을 주지만 정적 데이터로서의 몰입감은 한계를 가질 수밖에 없다. 이때, PBD를 사용하여 현실적인 물리 효과를 데이터에 적용하여 사용자에게 보다 현실적인 몰입감을 제공할 수 있다.

<table>
    <tbody>
    	<tr>
            <th style="text-align: center"><img src="https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/96eb5254-4ba1-40d6-a10f-d0858deeecc7" /></th>
        </tr>
    </tbody>
</table>


-픽킹

픽킹은 데이터에 직접적인 변형을 주는 부분으로 마우스 왼쪽 버튼을 클릭시 붉은 점이 생길 텐데 그 상태로 드래그하면 원하는 변형이 데이터 변형이 이뤄진다.

<table>
    <thead>
        <tr>
            <th style="text-align: center">Before</th>
        	<th style="text-align: center">After</th>
        </tr>
    </thead>
    <tbody>
    	<tr>
        	<th style="text-align: center"><img src="https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/89b1170d-9648-4a83-b717-fe15f97a9b96" alt="Before" style="zoom:80%;" /></th>
            <th style="text-align: center"><img src="https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/44e8560e-569b-49a0-a140-0bf6f428b84b" alt="After" style="zoom:80%;" /></th>
        </tr>
    </tbody>
</table>


## 기대효과
### 요구사항
- **Visual Studio 2022**
- **CUDA 12.4**
- **CMake 3.29.0**
- **Python 3.12**

### 리포지토리 클론 및 서브모듈 초기화
```shell
$ git clone --recursive https://github.com/GYEMOim/NeuralDynaRender
$ cd NeuralDynaRender
```

### 빌드 방법

Windows의 개발자 명령 프롬프트에서 다음 명령어를 실행한다:
CMake를 사용하여 프로젝트 빌드 설정:

```sh
$ cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

프로젝트 빌드:
```sh
$ cmake--build build --config RelWithDebInfo -j
```

### **nstant-ngp 시작 프로젝트 변경**

프로젝트 속성에서 디버깅 명령 인수 설정:`..data\nerf\fox` 



## 프로젝트 사용법

### 1. 촬영

대상이 되는 상대의 원하는 부분을 다양한 각도로 15초에서 25초 정도의 영상을 찍는다.

촬영한 영상은 NeuralDynaRender 파일 안에 있는 scripts 파일에 저장한다.

### 2. Python 스크립트 실행

scripts 폴더로 이동한 후, 변환하고 싶은 동영상을 폴더 안에 넣고(ex. demo.mp4) 아래 명령어로 Python 스크립트를 실행한다.

```sh
$ python3.12 colmap2nerf.py --video_in demo.mp4 --video_fps 2 --run_colmap --aabb_scale 2 --overwrite
```

영상의 프레임과 영상의 스케일은 사용자가 원하는 대로 설정하면 된다.

해당 명령어가 실행되면 폴더 내에 images 폴더와 transforms이라는 이름의 JSON 파일이 생성되었을 것이다.


### 3.프로그램 준비

scripts 위치에 있는 images 폴더와 JSON 파일을 data\nerf\에 폴더(ex. demo) 아래의 위치로 옮긴다.

```sh
NeuralDynaRender\data\nerf\demo
```
이때 demo 폴더는 사용자가 생성한 폴더의 이름이다.

### 4.프로그램 실행
cmd의 위치를 NeuralDynaRender로 옮겨준 뒤, 아래의 명령어를 통해 프로그램을 실행합니다.

```sh
NeuralDynaRender$ instant-ngp data/nerf/demo
```

### 학습 및 변형
프로그램이 실행되면 아래와 같은 화면이 뜰 것이다.

![Untitled](https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/45d397fa-17cd-4f6c-8ff5-f4733286701b)

1분에서 2분 사이를 기다려 프로그램이 데이터를 학습할 시간을 마련한다. 이후, 더 이상의 학습을 원하지 않는 경우 stop training을 선택하여 학습을 멈춘다.

Open Rendering Option을 클릭하면 아래와 같은 창이 추가로 뜰 것이다.

![image](https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/84b9dbb3-9a54-4b7a-8064-64947e5b172f)

원하는 모드를 선택한 뒤, RGBA.den를 클릭한다.

![image](https://github.com/GYEMOim/NeuralDynaRender/assets/100848728/135097c0-79ff-4d8a-8603-5c0a080b98f9)

위와 같은 렌더링 화면이 성공적으로 출력되면 왼쪽 마우스 버튼으로 시점 변환을 할 수 있고, 휠로 확대와 축소를 할 수 있다.

오른쪽 마우스 버튼 클릭 시, 붉은 점으로 선택한 데이터의 위치가 표시될 것이고 드래그와 같은 행위를 통해 데이터 변형을 시도할 수 있다.



