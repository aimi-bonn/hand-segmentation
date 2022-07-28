"""
utility funcs to handle encapsulated objects
"""
import numpy as np


def handle_encapsulated_contour(outer_cont, inner_cont) -> np.ndarray:
    """
    given an outer contour (potentially) containing included inner contours this
    functions generates a contour representing the exact topology (outer contour
    without in the inner contours).

    Args:
        outer_cont (np.ndarray): Outer contour obtained using opencv
        inner_cont (list): list containing inner contours. If empty, this functions
         returns the unchanged outer_cont

    Returns:
         topology-corrected outer contour
    """
    if len(inner_cont) == 0:
        return outer_cont
    assert (
        inner_cont[0].shape[1] == 1 and inner_cont[0].shape[2] == 2
    ), "wrong contour format provided, please use raw cv2 output"
    joint = [outer_cont]  # outer obj
    end_point = outer_cont[-1].reshape(1, 1, 2)  # end point of outer cont
    for incl_obj in inner_cont:
        joint.append(incl_obj)  # walk inner object
        joint.append(incl_obj[0].reshape(1, 1, 2))  # go to start point of inner cont
        joint.append(end_point)  # close inner cont by going back to outer end point
    joint = np.vstack(joint)
    return joint


def handle_contour_topology(contours, hierarchy, contour_index=0) -> np.ndarray:
    """
    generates a contour excluding encapsulated holes from cv2.findContours() results
    (deeper levels of topology are ignored, see demo)

    Args:
        contours (list[np.ndarray]): contours obtained from cv2.findContours()
        hierarchy (np.ndarray): contour hierarchy obtained from cv2.findContours()
        contour_index (int): topology-corrected outer contour (contour excluding holes)

    Returns:
          Corrected contour
    """
    assert len(contours) != 0
    assert (
        contours[0].shape[1] == 1 and contours[0].shape[2] == 2
    ), "wrong contour format provided, please use raw cv2 output"
    assert (
        hierarchy.shape[0] == 1 and hierarchy.shape[2] == 4
    ), "wrong hierarchy format provided, please use raw cv2 output"

    child = hierarchy[0, contour_index, 2]  # get first child
    if child == -1:  # no child -> nothing to do
        return contours[contour_index]
    else:
        encaps_objects = [contours[child]]
        while (
            hierarchy[0, child, 0] != -1
        ):  # while there is another contour on same (e.g. inner/child) hierarchy  level
            child = hierarchy[0, child, 0]
            encaps_objects.append(contours[child])
        return handle_encapsulated_contour(contours[contour_index], encaps_objects)
