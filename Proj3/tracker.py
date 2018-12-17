# Short tracker-tool, should display the progress as a rising bar.
# For implementation on backend to keep showing the progress as the 
# longer ones take place. 
import sys
from time import sleep

sys.stdout.write('['+' '*20 +']')
for i in range(21):
    sys.stdout.write('\r[\033[48;5;15m') 
    prog = " "*i
    sys.stdout.write(prog)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()
    sleep(0.01)
sys.stdout.write('\n')


N = 100

for i in range(N+1):
    # Code block
    percentage = int((float(i)/N) * 50) #-ish.
    if percentage%2 == 0: 
        remaining_percentage = str( " "*(50 - percentage)+"]")
        percentage_treat = "[\033[48;5;15m" +" "*percentage + "\033[0m"
        string = "Iteration %4d progress:"%i 
        sys.stdout.write("\r%s %s%s"%(string,percentage_treat, remaining_percentage) )
        sys.stdout.flush()
    sleep(0.05)
sys.stdout.write('Done!\n')
