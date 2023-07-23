from django.shortcuts import render

def prediction_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        # TODO: model predict
        # prediction = 

        prediction = user_input

        context = {
            'prediction': prediction,
            'user_input': user_input
        }

        return render(request, 'prediction.html', context)
    
    return render(request, 'index.html')
