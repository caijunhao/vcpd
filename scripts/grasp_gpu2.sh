num=2280
for ((i=1; i<=$num; i=i+3))
do
 python gripper_grasp.py --obj_idx $i --gpu 3
done
