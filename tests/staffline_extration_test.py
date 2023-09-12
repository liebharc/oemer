from oemer.staffline_extraction import Staff, align_row
from pytest import approx

def create_staff(x_left, y_upper, x_right, y_lower):
    staff = Staff()
    staff.x_left = x_left
    staff.y_upper = y_upper
    staff.x_right = x_right
    staff.y_lower = y_lower
    staff.x_center = (x_left + x_right) / 2
    staff.y_center = (y_upper + y_lower) / 2
    return staff

def test_align_row():
    line = [create_staff(37, 712, 222, 782), create_staff(223, 729, 427, 804), create_staff(428, 712, 632, 782), create_staff(633, 712, 837, 782), create_staff(838, 712, 1042, 782), create_staff(1043, 711, 1247, 782), create_staff(1248, 711, 1452, 783), create_staff(1453, 711, 1648, 783)]
    result = align_row(line)
    expected = list(line)
    expected[1] = create_staff(223, 712, 427, 782)
    assert [staff.y_center for staff in result] == approx([staff.y_center for staff in expected], abs=0.1)
