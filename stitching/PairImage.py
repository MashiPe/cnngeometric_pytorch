from stitching.Image import Image

class PairImage:

    def __init__(self,
                imgTgt: Image,
                imgSrc: Image,
                y_src_coord: int,
                x_src_coord: int,
                x_src_dir: int ):

        self.imgTgt = imgTgt
        self.imgSrc = imgSrc
        self.x_src_coord = x_src_coord
        self.y_src_coord = y_src_coord
        self.x_src_dir = x_src_dir

