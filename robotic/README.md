# Robotic experiment instructions

1. Be sure you already installed all libraries required in ext_theory. 
2. Install dependencies
   ```
   cd ext_theory/robotic/scripts
   pip install -r requirements.txt
   ```
3. Add environment path
   ```
   export PYTHONPATH=/path/to/ext_theory/robotic/BulletArm:$PYTHONPATH
   ```
4. Run harmful extrinsic equivariance experiment
    ```
    # harmful extrinsic equivariant using flip equivariance
    python main.py --env=close_loop_block_color_sort --model=equi_d1
    # cnn baseline
    python main.py --env=close_loop_block_color_sort --model=cnn
    ```
5. Run incorrect equivariance experiment
   * adjust --correct_ratio argument to change the ratio, e.g., --correct_ratio=0.6 means 60% correct data and 40% incorrect data 
    ```
    cd ext_theory/robotic/scripts
    # incorrect equivariant using flip equivariance
    python main.py --env=close_loop_block_arranging --model=equi_d1 --correct_ratio=0.6
    # cnn baseline
    python main.py --env=close_loop_block_arranging --model=cnn --correct_ratio=0.6
    ```
## Note
* Add --render=t argument to view real-time visualization
