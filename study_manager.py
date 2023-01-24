import json

def write_screen(status):

    print('\n'*40)
    print('***************')
    print(f'Location : {status["location"]}')
    print('\n'*1)
    
    print('Options:')
    for i, option in enumerate(status['options']):
        print(f'({i}) {option}')
        
    print('\n'*1)
    print('***************')
    
    selection = int(input('Navigate to which view : '))
    assert( selection in range(len(status['options'])))

    return selection

class study_manager():

    def __init__(self, ):
        
        print('\nWelcome back!\n')
        self.load_brain()
        self.run()
        
    def load_brain(self, ):
        with open('smb.json') as file:
            brain = json.load(file)
        
        #validate read
        self.smb = brain
        return
    
    def save_brain(self, ):
        #validate save
        with open('smb.json', 'w') as outfile:
            json.dump(self.smb, outfile)
        return
        
    def run(self,):
    
        status = {
            'location' : 'home',
            'options' : ['exit', 'second', 'third']      
        }
        selection = True
    
    
        while selection != 0 :
        
            selection = write_screen(status)
            
    
        return
    

sm = study_manager()


