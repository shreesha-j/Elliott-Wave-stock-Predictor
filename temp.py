# import threading
# import time

# def time_consuming_task():
#     # Simulating a time-consuming task
#     time.sleep(5)

# def display_loader():
#     while thread.is_alive():
#         loader = ['|', '/', '-', '\\']
#         for char in loader:
#             print(f"\rLoading... {char}", end='', flush=True)
#             time.sleep(0.1)

# # Create a thread for the time-consuming task
# thread = threading.Thread(target=time_consuming_task)

# # Start the thread
# thread.start()

# # Display the loader
# display_loader()

# # Wait for the thread to finish
# thread.join()

# # Execution continues here after the time-consuming task is completed
# print("\nTask completed!")

# import itertools
# import threading
# import time
# import sys

# done = False
# #here is the animation
# def animate():
#     for c in itertools.cycle(['|', '/', '-', '\\']):
#         if done:
#             break
#         sys.stdout.write('\rloading ' + c)
#         sys.stdout.flush()
#         time.sleep(0.1)
#     sys.stdout.write('\rDone!     ')

# t = threading.Thread(target=animate)
# t.start()

# #long process here
# time.sleep(10)
# done = True

import streamlit as st 
e = RuntimeError('This is an exception of type RuntimeError')
st.exception(e)