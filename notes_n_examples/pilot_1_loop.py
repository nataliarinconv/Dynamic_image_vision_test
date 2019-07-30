import natalialibNew as nlib;
import getch;
import time;

pilot_1 = nlib.Natalia_object();
continue_vs_not = 'Press 1 to continue press 2 to end: '
skip_convolution = 0;

while True:
	try:
		print(continue_vs_not);
		loop = int(getch.getch());
		if loop == 1:
			print('run_block');
			if skip_convolution == 1:
				nlib.run_block(pilot_1,skip_convolution);			
			elif skip_convolution == 0:
				nlib.run_block(pilot_1,skip_convolution);			
				skip_convolution = 1;
		elif loop == 2:
			print('You have exited loop. Bye bye!');
			break
		else:
			raise ValueError;
	except ValueError:
		print("Not 1 or 2, try again");
		continue;

