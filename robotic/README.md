# Robotic experiment instructions

1. Be sure you already installed all libraries required by swiss_roll. 
1. Clone the environment repo
    ```
    git clone https://github.com/SaulBatman/BulletArm.git -b ext_equi
    export PYTHONPATH=/path/to/BulletArm/:$PYTHONPATH
    ```
3. Install dependencies
   ```
   cd swiss_roll/robotic/scripts
   pip install -r requirements.txt
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
    cd swiss_roll/robotic/scripts
    # incorrect equivariant using flip equivariance
    python main.py --env=close_loop_block_arranging --model=equi_d1 --correct_ratio=0.6
    # cnn baseline
    python main.py --env=close_loop_block_arranging --model=cnn --correct_ratio=0.6
    ```
## Note
* Add --render=t argument to view real-time visualization