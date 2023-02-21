
using System;
using OpenCvSharp;


namespace KaleidoscopeApp
{
    public class Program
    {
        static void Main(string[] args)
        {
            VideoCapture vidSrc = openVidSource();
           
            double frameRate;
            if (vidSrc.CaptureType == CaptureType.Camera)
                frameRate = 25.0;
            else
                frameRate = vidSrc.Fps;

            string fname = inputResFilename();
            Console.WriteLine("Sample size: ");
            int size = int.Parse(Console.ReadLine());
            string window = "Kaleidoscope";
            Cv2.NamedWindow(window, WindowFlags.FullScreen);
            Cv2.ResizeWindow(window, 1920, 1080);
            Cv2.MoveWindow(window, 0, 0);

            var winRect = Cv2.GetWindowImageRect(window);
            int width = Math.Min(winRect.Height, size);
            int height = (int)(Math.Sqrt(3) / 2 * width);
            
            Mat dst =  new Mat(winRect.Height / height * height, winRect.Width / width * width, MatType.CV_8UC3, new Scalar(0, 0, 0));
            Mat mapX = new Mat(winRect.Height / height * height, winRect.Width / width * width, MatType.CV_32FC1, new Scalar(0));
            Mat mapY = new Mat(winRect.Height / height * height, winRect.Width / width * width, MatType.CV_32FC1, new Scalar(0));
            Kaleidoscope.GenerateKaleidoscopeMap(ref mapX, ref mapY, width, height);

            VideoWriter vidRes = null;

            if (fname != null)
            {
                vidRes = new VideoWriter(fname, FourCC.Default, frameRate, new Size(dst.Width, dst.Height));
                if (!vidRes.IsOpened())
                {
                    Console.WriteLine("Couldn't open file!");
                }
            }
            Console.WriteLine("Press any key to exit.");
            Mat frame = new Mat();
            vidSrc.Read(frame);
            Mat src = new Mat(frame, new OpenCvSharp.Range(frame.Height / 2 - height / 2 - 1, frame.Height / 2 + height / 2 + 1), new OpenCvSharp.Range(frame.Width / 2 - width / 2 - 1, frame.Width / 2 + width / 2 + 1));

            while (true)
            {
                vidSrc.Read(frame);
                if (frame.Empty())
                    break;
                Cv2.Remap(src, dst, mapX, mapY);
                Cv2.ImShow("Kaleidoscope", dst);
                vidRes?.Write(dst);
                if (Cv2.WaitKey((int)(1000 / frameRate)) != -1)
                    break;
            }
            vidSrc.Release();
            vidRes?.Release();
        }

        static VideoCapture openVidSource()
        {
            while (true)
            {
                VideoCapture vidSrc = new VideoCapture();
                Console.WriteLine("Video source: (C)amera, (F)ile");
                string input = Console.ReadLine();
                switch (input[0])
                {
                    case 'c':
                    case 'C':
                        Console.WriteLine("Device to use (0 = default): ");
                        int deviceID = int.Parse(Console.ReadLine());
                        vidSrc.Open(deviceID);
                        if (!vidSrc.IsOpened())
                        {
                            Console.WriteLine("Couldn't open camera!");
                            break;
                        }
                        return vidSrc;
                    case 'f':
                    case 'F':
                        Console.WriteLine("Filename: ");
                        string fname = Console.ReadLine();
                        vidSrc.Open(fname);
                        if (!vidSrc.IsOpened())
                        {
                            Console.WriteLine("Couldn't open file!");
                            break;
                        }
                        return vidSrc;
                    default:
                        Console.WriteLine("Invalid input!");
                        break;
                }
            }
        }
        static string inputResFilename()
        {
            while (true)
            {
                Console.WriteLine("Write output to file? (y/n)");
                string input = Console.ReadLine();
                switch (input[0])
                {
                    case 'y':
                    case 'Y':
                        Console.WriteLine("Output file name (with extension): ");
                        string fname = Console.ReadLine();
                        return fname;
                    case 'n':
                    case 'N':
                        return null;
                    default:
                        Console.WriteLine("Invalid input!");
                        break;
                }
            }
        }
    }
}