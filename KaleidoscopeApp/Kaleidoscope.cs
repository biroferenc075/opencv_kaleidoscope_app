using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace KaleidoscopeApp
{
    public class Kaleidoscope
    {
        /// <summary>
        /// Generates the submaps for the sections of the image with the given width and height, 
        /// [0:2] are facing downwards,
        /// [3:5] are upwards.
        /// </summary>
        public static List<(Mat mapX, Mat mapY)> GenerateSubMaps(int width, int height)
        {
            var subMaps = new List<(Mat, Mat)>();
            Mat originalX = new Mat(new Size(width, height), MatType.CV_32FC1, new Scalar(0));
            Mat originalY = new Mat(new Size(width, height), MatType.CV_32FC1, new Scalar(0));
            var oIndX = originalX.GetGenericIndexer<float>();
            var oIndY = originalY.GetGenericIndexer<float>();

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    oIndX[y, x] = x;
                    oIndY[y, x] = y;
                }
            }
            for (int n = 0; n < 3; n++)
            {
                Mat mapX = new Mat(new Size(width, height), MatType.CV_32FC1, new Scalar(0));
                Mat mapY = new Mat(new Size(width, height), MatType.CV_32FC1, new Scalar(0));
                var Rot = Cv2.GetRotationMatrix2D(new Point2f(width / 2.0f, height / 3.0f), -120.0 * (2 * n - 1), 1);
                Cv2.WarpAffine(originalX, mapX, Rot, new Size(width, height));
                Cv2.WarpAffine(originalY, mapY, Rot, new Size(width, height));
                subMaps.Add((mapX, mapY));
            }
            for (int n = 0; n < 3; n++)
            {
                Mat mapX = new Mat(new Size(width, height), MatType.CV_32FC1, new Scalar(0));
                Mat mapY = new Mat(new Size(width, height), MatType.CV_32FC1, new Scalar(0));
                var Rot = Cv2.GetRotationMatrix2D(new Point2f(width / 2.0f, height / 3.0f), 120.0 * n, 1);
                Cv2.WarpAffine(originalX, mapX, Rot, new Size(width, height));
                Cv2.WarpAffine(originalY, mapY, Rot, new Size(width, height));
                Cv2.Flip(mapX, mapX, FlipMode.X);
                Cv2.Flip(mapY, mapY, FlipMode.X);
                subMaps.Add((mapX, mapY));
            }
            return subMaps;
        }
        /// <summary>
        /// Generates the triangle shaped masks with the given width and height,
        /// [0] is pointing downwards, [1] upwards
        /// </summary>
        public static List<Mat> GenerateMasks(int width, int height)
        {
            var masks = new List<Mat>();
            Mat maskPosY = new Mat(new Size(width, height), MatType.CV_8UC1, new Scalar(0));
            Mat maskNegY = new Mat(new Size(width, height), MatType.CV_8UC1, new Scalar(0));
            var ind = maskNegY.GetGenericIndexer<byte>();

            for (int x = 0; x < width / 2; x++)
            {
                float ymax = x * 2 * (float)height / width;
                for (int y = 0; y < ymax; y++)
                {
                    ind[y, x] = 255;
                }
            }
            for (int x = width / 2; x < width; x++)
            {
                float ymax = height - (x - width / 2) * 2 * (float)height / width;
                for (int y = 0; y < ymax; y++)
                {
                    ind[y, x] = 255;
                }
            }
            Cv2.Flip(maskNegY, maskPosY, FlipMode.X);

            masks.Add(maskNegY);
            masks.Add(maskPosY);
            return masks;
        }
        /// <summary>
        /// Generates the maps used with the Remap funcion,
        /// with resX being the map for the X coords, 
        /// and resY for the Y coordinates,
        /// the width and height parameters as the sizes of the triangular subsections.
        /// </summary>
        public static void GenerateKaleidoscopeMap(ref Mat resX, ref Mat resY, int width, int height)
        {
            var submaps = GenerateSubMaps(width, height);
            var masks = GenerateMasks(width, height);
            var imWidth = resX.Width;
            var cols = resX.Width / width;
            var rows = resX.Height / height;
            Mat xRoi;
            Mat yRoi;
            Rect roi;

            // Copy upwards facing sections
            roi = new Rect(0, 0, width, height);
            xRoi = resX.SubMat(roi);
            yRoi = resY.SubMat(roi);

            for (int n = 0; n < cols; n++)
            {
                submaps[n % 3].mapX.CopyTo(xRoi, masks[0]);
                submaps[n % 3].mapY.CopyTo(yRoi, masks[0]);
                xRoi.AdjustROI(0, 0, -width, width);
                yRoi.AdjustROI(0, 0, -width, width);
            }

            // Copy downwards facing sections
            roi = new Rect(width / 2, 0, width, height);
            xRoi = resX.SubMat(roi);
            yRoi = resY.SubMat(roi);
            for (int n = 0; n < cols - 1; n++)
            {
                submaps[n % 3 + 3].mapX.CopyTo(xRoi, masks[1]);
                submaps[n % 3 + 3].mapY.CopyTo(yRoi, masks[1]);
                xRoi.AdjustROI(0, 0, -width, width);
                yRoi.AdjustROI(0, 0, -width, width);
            }

            // Copy half-section on the left edge
            roi = new Rect(0, 0, width / 2, height);
            xRoi = resX.SubMat(roi);
            yRoi = resY.SubMat(roi);

            var subMatRoi = new Rect(width / 2, 0, width / 2, height);
            var edgeRoiX = submaps[5].mapX.SubMat(subMatRoi);
            var edgeRoiY = submaps[5].mapY.SubMat(subMatRoi);
            var maskRoi = masks[1].SubMat(subMatRoi);
            edgeRoiX.CopyTo(xRoi, maskRoi);
            edgeRoiY.CopyTo(yRoi, maskRoi);

            // Copy half-section on the right edge
            roi = new Rect(imWidth - width / 2, 0, width / 2, height);
            xRoi = resX.SubMat(roi);
            yRoi = resY.SubMat(roi);

            subMatRoi = new Rect(0, 0, width / 2, height);
            edgeRoiX = submaps[(cols - 1) % 3 + 3].mapX.SubMat(subMatRoi);
            edgeRoiY = submaps[(cols - 1) % 3 + 3].mapY.SubMat(subMatRoi);
            maskRoi = masks[1].SubMat(subMatRoi);
            edgeRoiX.CopyTo(xRoi, maskRoi);
            edgeRoiY.CopyTo(yRoi, maskRoi);

            // Fill in the rest of the rows with the mirror image of the previous one
            var indexerX = resX.GetGenericIndexer<float>();
            var indexerY = resY.GetGenericIndexer<float>();
            for (int n = 1; n < rows; n++)
            {
                for (int x = 0; x < imWidth; x++)
                    for (int y = 0; y < height; y++)
                    {
                        indexerX[y + n * height, x] = indexerX[height - y + (n - 1) * height - n, x];
                        indexerY[y + n * height, x] = indexerY[height - y + (n - 1) * height - n, x];
                    }
            }
        }
    }
}
