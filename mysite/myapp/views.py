from django.shortcuts import render
from .model_utils import model_prediction

def prediction_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')

        prediction_rating, prediction_emotion = model_prediction(user_input=user_input) 
        if prediction_emotion == 1:
            emotion = 'Positive'
        else:
            emotion = 'Negative'
        
        if prediction_rating < 4:
            rating = f'{prediction_rating+1} Stars'
        else:
            rating = f'{prediction_rating+3} Stars'
        
        context = {
            'prediction_rating': rating,
            'prediction_emotion': emotion,
            'user_input': user_input
        }

        return render(request, 'prediction.html', context)
    
    return render(request, 'index.html')
