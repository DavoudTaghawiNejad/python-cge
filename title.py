length = 100

def title(titel, style='+'):
    print('')
    sides = ''.join([style] *  (((length - len(titel)) - 4 ) / 2))
    middle =  sides + '  ' + titel + '  ' + sides
    print(''.join([style] * len(middle)))
    print(middle)
    print(''.join([style] * len(middle)))
    
def heading(text):
    title(text, '=')
