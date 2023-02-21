using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using KaleidoscopeApp;

namespace Testing
{
    public class TestKaleidoscope
    {
        private Mat GenerateTestImage(int width, int height)
        {
            // bottom half is red
            var image = new Mat(height, width, MatType.CV_8UC3, new Scalar(0, 0, 255));
            
            var indexer = image.GetGenericIndexer<Vec3b>();
            // top half is blue
            for (int y = 0; y < height / 3; y++)
                for (int x = 0; x < width; x++)
                    indexer[y, x] = new Vec3b(255, 0, 0);
            return image;
        }
        [Fact]
        public void SubmapTest()
        {
            int width = 100;
            int height = (int)(100 * Math.Sqrt(3) / 2);
            var submaps = Kaleidoscope.GenerateSubMaps(width, height);
            var img = GenerateTestImage(width, height);
            List<Vec3b> expectedValues = new List<Vec3b>();
            expectedValues.Add(new Vec3b(0, 0, 255)); // rotated 120°
            expectedValues.Add(new Vec3b(0, 0, 255)); // rotated 240°
            expectedValues.Add(new Vec3b(255, 0, 0)); // original image
            expectedValues.Add(new Vec3b(255, 0, 0)); // flipped
            expectedValues.Add(new Vec3b(0, 0, 255)); // flipped and rotated 120°
            expectedValues.Add(new Vec3b(0, 0, 255)); // flipped and rotated 240°

            for(int i = 0; i < submaps.Count; i++)
            {
                var submap = submaps[i];
                var res = new Mat(new Size(width, height),MatType.CV_8UC3);
                Cv2.Remap(img, res, submap.mapX, submap.mapY);
                var indexer = res.GetGenericIndexer<Vec3b>();
                if(i < 3) // upwards oriented submaps
                    Assert.Equal(expectedValues[i], indexer[10, width / 2]); // color around top vertex 
                if (i >= 3) // downwards oriented submaps
                    Assert.Equal(expectedValues[i], indexer[height-10, width / 2]); // color around bottom vertex
            }
        }
        [Fact]
        public void MaskTest()
        {
            int width = 100;
            int height = (int)(100 * Math.Sqrt(3) / 2);
            var masks = Kaleidoscope.GenerateMasks(width, height);
            var indexer = masks[0].GetGenericIndexer<byte>();
            Assert.Equal(255, indexer[5, 10]); // top left vertex
            Assert.Equal(255, indexer[height/2-5, width/2]); // bottom vertex
            Assert.Equal(255, indexer[5, width-10]); // top right vertex
            Assert.Equal(0, indexer[height-5, 10]); // bottom left corner of image
            Assert.Equal(0, indexer[height-5, width-10]); // bottom right corner of image

            indexer = masks[1].GetGenericIndexer<byte>();
            Assert.Equal(255, indexer[height-5, 10]); // top left vertex
            Assert.Equal(255, indexer[5, width / 2]); // bottom vertex
            Assert.Equal(255, indexer[height-5, width - 10]); // top right vertex
            Assert.Equal(0, indexer[5, 10]); // bottom left corner of image
            Assert.Equal(0, indexer[5, width - 10]); // bottom right corner of image
        }
        [Fact]
        public void KaleidoscopeTest()
        {
            int width = 100;
            int height = (int)(100 * Math.Sqrt(3) / 2);
            Mat img = GenerateTestImage(width, height);
            Mat dst =  new Mat(3 * height, 3 * width, MatType.CV_8UC3, new Scalar(0, 0, 0));
            Mat mapX = new Mat(3 * height, 3 * width, MatType.CV_32FC1, new Scalar(0));
            Mat mapY = new Mat(3 * height, 3 * width, MatType.CV_32FC1, new Scalar(0));
            Kaleidoscope.GenerateKaleidoscopeMap(ref mapX, ref mapY, width, height);
            Cv2.Remap(img, dst, mapX, mapY);
            Vec3b red = new Vec3b(0, 0, 255);
            Vec3b blue = new Vec3b(255, 0, 0);

            List<Point> points = new List<Point>();
            List<Vec3b> expectedColors = new List<Vec3b>();
            var indexer = dst.GetGenericIndexer<Vec3b>();
            points.Add(new Point(5, 5));
            expectedColors.Add(blue);
            points.Add(new Point(2*width, 5));
            expectedColors.Add(blue);
            points.Add(new Point(3*width-5, 5));
            expectedColors.Add(blue);
            points.Add(new Point(5, 2*height));
            expectedColors.Add(blue);
            points.Add(new Point(2*width, 2 * height));
            expectedColors.Add(blue);

            points.Add(new Point(5, height));
            expectedColors.Add(red);
            points.Add(new Point(width, height/2));
            expectedColors.Add(red);
            points.Add(new Point(width*2+width/2, height));
            expectedColors.Add(red);
            points.Add(new Point(width, height * 2));
            expectedColors.Add(red);
            points.Add(new Point(width * 2 + width / 2, 3*height-5));
            expectedColors.Add(red);

            for (int i = 0; i < points.Count; i++)
                Assert.Equal(expectedColors[i], indexer[points[i].Y, points[i].X]);
        }
    }

    
}
