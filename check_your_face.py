def check_your_face(eye, cheek):
    if eye < 0 or cheek < 0:
        eye *= -1
        cheek *= -1
        
    if eye <= 1 and cheek <= 1:
        print("당신은 좌우가 99.9% 일치합니다")
    elif eye <= 6 and cheek <= 10:
        print("눈과 볼의 대칭이 아주 환상적이네요")
    elif eye <= 10 and cheek <= 15:
        print("평균입니다!")
    else:
        print("다시 태어나십시요!")
