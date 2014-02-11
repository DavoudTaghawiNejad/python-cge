length = 100

def general(text, style='-', extra=0):
    sides = ''.join([style] *  (((length - len(text)) - 4 ) / 2))
    titel =  sides + '  ' + text + '  ' + sides
    middle = sides + ''.join([' '] * (len(text) + 4)) + sides
    print('')
    for _ in range(extra):
        print(''.join([style] * len(middle)))
    for _ in range(extra):
        print(middle)
    print(titel)
    for _ in range(extra):
        print(middle)
    for _ in range(extra):
        print(''.join([style] * len(middle)))
    print('')

def heading(text):
    general(text, '=', extra=1)

def title(text, style='+'):
    for _ in range(100):
        print('')
    general(text, '+', 2)
    for _ in range(5):
        print('')