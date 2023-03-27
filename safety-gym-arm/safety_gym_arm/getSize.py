import mujoco_py
import xml.etree.ElementTree as ET
# Load the Mujoco model and simulate it
# model = mujoco_py.load_model_from_path('xmls/arm.xml')
# sim = mujoco_py.MjSim(model)

# Get the ID of the geometry for body named 'my_body'
# geom_names = [model.geom_names[i] for i in range(model.ngeom)]
# print(geom_names)

# for i in range(len(geom_names)):
#     # Find the index of the geom with the specified name
#     geom_id = model.geom_names.index(geom_names[i])
#     # geom_id2 = model.body_name2id(geom_names[i])
#     # capsule_size = mujoco_py.functions.mj_obj_getSize(sim.model, sim.data, geom_id)
#     print(geom_id)
#     geom_size = model.geom_size[geom_id]
#     print(geom_names[i])
#     print("Size ': {}".format(geom_size))