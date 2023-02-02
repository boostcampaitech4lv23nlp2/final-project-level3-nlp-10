import json

def convert2context(json_path):
    with open(json_path, 'r') as f:
        speech = json.load(f)
    # result만 있고 없고 분기
    last_label = -1
    context = []
    text = ''
    for segment in speech['segments']:
        label = segment['speaker']['label']
        if last_label == label:
            text += " " + segment['textEdited']
        else:
            context.append(text)
            text = segment['textEdited']
            last_label = label
    return context
        

    