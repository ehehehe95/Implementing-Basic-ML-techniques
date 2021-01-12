package com.yooong.studynotefragment;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.fragment.app.DialogFragment;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;

import com.google.android.gms.ads.AdRequest;
import com.google.android.gms.ads.AdView;
import com.google.android.gms.ads.MobileAds;
import com.google.android.gms.ads.initialization.InitializationStatus;
import com.google.android.gms.ads.initialization.OnInitializationCompleteListener;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.gun0912.tedpermission.PermissionListener;
import com.gun0912.tedpermission.TedPermission;
import com.yalantis.ucrop.UCrop;
import com.yooong.studynotefragment.env.Logger;
import com.yooong.studynotefragment.env.Utils;
import com.yooong.studynotefragment.tflite.Classifier;
import com.yooong.studynotefragment.tflite.YoloV4Classifier;
import com.yooong.studynotefragment.utils.UiHelper;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements FolderFragment.folderClickListener, FolderDialogFragment.dialogListener {
    public static File dir;
    public static int fragmentPage;
    private FloatingActionButton gallery, camera, add_button, aiSettingButton;
    private SwitchMaterial aiSwitch;
    public static File problemNote;

    public static final int CAMERA_ACTION_PICK_REQUEST_CODE = 610;
    public static final int PICK_IMAGE_GALLERY_REQUEST_CODE = 609;

    public static final int CAMERA_STORAGE_REQUEST_CODE = 611;
    public static final int ONLY_CAMERA_REQUEST_CODE = 612;
    public static final int ONLY_STORAGE_REQUEST_CODE = 613;

    private String currentPhotoPath = "";

    private UiHelper uiHelper = new UiHelper();


    private String folderName;
    private int subjectId;

    private File destinationFile;

    //TensorFLow

    private Classifier detector;

    boolean aiFlag = false;
    public static boolean onlyWrong = true;

    public static float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

    private static final Logger LOGGER = new Logger();

    public static final int TF_OD_API_INPUT_SIZE = 416;

    private static final boolean TF_OD_API_IS_QUANTIZED = false;

    private static final String TF_OD_API_MODEL_FILE = "yolo4-3000-tiny.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/problem.txt";

    private AdView mAdView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        //admob
        MobileAds.initialize(this, new OnInitializationCompleteListener() {
            @Override
            public void onInitializationComplete(InitializationStatus initializationStatus) {
            }
        });

        mAdView = findViewById(R.id.adView);
        AdRequest adRequest = new AdRequest.Builder().build();
        mAdView.loadAd(adRequest);



        //init detector
        try {
            detector =
                    YoloV4Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED);
        } catch (final Exception e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        //권한 체크
        TedPermission.with(getApplicationContext())
                .setPermissionListener(permissionListener)
                .setDeniedMessage("설정에서 권한을 허용해 주세요")
                .setPermissions(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                .check();

        //make studyNoteFolder in gallery

        File file = this.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        Log.d("test1", file.getPath());

        dir = new File(file.getAbsoluteFile() + File.separator + "studyNote");
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                Log.d("makeStudyNote", "failed");
            } else {
                Log.d("makeStudyNote", "Success");
            }

        } else {
            Log.d("makeStudyNote", "alreadyExists");
        }

//        FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
//        Fragment folderFragment = new FolderFragment();
//        transaction.replace(R.id.frame,folderFragment);
//        transaction.addToBackStack(null);
//        transaction.commit();
        rootFolder();

        //Button actions
        add_button = (FloatingActionButton) findViewById(R.id.button_parent);
        gallery = (FloatingActionButton) findViewById(R.id.addFromGallery);
        camera = (FloatingActionButton) findViewById(R.id.addFromCamera);
        aiSettingButton = findViewById(R.id.aiSetting);

        aiSettingButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                DialogFragment aiDialogFragment = new AiDialogFragment();
                aiDialogFragment.show(getSupportFragmentManager(), "aiDialogFragment");

            }
        });

        add_button.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {

                if (fragmentPage == 0) {
                    DialogFragment folderDialogFragment = new FolderDialogFragment();
                    Log.d("check", "folderDialogmade");
                    folderDialogFragment.show(getSupportFragmentManager(), "folderDialogFragment");
                } else {

                    gallery.setVisibility(View.VISIBLE);
                    camera.setVisibility(View.VISIBLE);

                }

            }
        });


        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //openGallery();
                gallery.setVisibility(View.GONE);
                camera.setVisibility(View.GONE);
                openImagesDocument();
            }
        });


        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    gallery.setVisibility(View.GONE);
                    camera.setVisibility(View.GONE);
                    Log.d("crush_test2","openCameraStart");
                    openCamera();
                } catch (SecurityException e) {
                    Log.d("crush_test2","security Exception"+e.getMessage());
                    TedPermission.with(getApplicationContext())
                            .setPermissionListener(permissionListener)
                            .setDeniedMessage("설정에서 권한을 허용해 주세요")
                            .setPermissions(Manifest.permission.CAMERA)
                            .check();
                }
            }
        });

        aiSwitch = (SwitchMaterial) findViewById(R.id.aiSwitch);
        aiFlag = aiSwitch.isChecked();
        aiSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                aiFlag = isChecked;
            }
        });
    }

    public void rootFolder() {
        FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
        Fragment folderFragment = new FolderFragment();
        transaction.replace(R.id.frame, folderFragment, "folderFragment");
        transaction.addToBackStack(null);
        transaction.commit();
    }

//    public void openFolder(String folderName){
//        problemNote =  new File(dir.getAbsolutePath()+File.separator+folderName);
//
//        ArrayList<String> imagePathList = new ArrayList<>();
//        if(problemNote.exists()) {
//            File[] allFiles = problemNote.listFiles(new FilenameFilter() {
//                public boolean accept(File dir, String name) {
//                    return (name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png"));
//                }
//            });
//            for(File f : allFiles){
//                imagePathList.add(f.getAbsolutePath());
//            }
//
//        }
//
//        FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
//        Log.d("check","transcation start");
//        Fragment galleryFragment = new GalleryFragment(imagePathList);
//        Log.d("check","galleryFragmentMade");
//        transaction.replace(R.id.frame,galleryFragment,"galleryFragment");
//        transaction.addToBackStack(null);
//        transaction.commit();
//
//    }

    //Permission

    PermissionListener permissionListener = new PermissionListener() {
        @Override
        public void onPermissionGranted() {
        }

        @Override
        public void onPermissionDenied(ArrayList<String> deniedPermissions) {
            Toast.makeText(getApplicationContext(), "앱을 사용하기 위해서는 권한 허용이 필요합니다!", Toast.LENGTH_SHORT).show();
        }
    };

    // crop & camera & album

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // Succed taking picture from Camera
        if (requestCode == CAMERA_ACTION_PICK_REQUEST_CODE && resultCode == RESULT_OK) {
            Uri uri = Uri.parse(currentPhotoPath);
//            Uri test = Uri.fromFile(new File(problemNote,""+System.currentTimeMillis()));
//            Log.d("test",test.getPath());
            destinationFile = new File(problemNote, System.currentTimeMillis() + ".jpg");
            if (aiFlag == true) {
                openAiCameraCropActivity(uri, uri.fromFile(destinationFile));
            } else {
                openCropActivity(uri, Uri.fromFile(destinationFile));
            }

        }
        //Succeed cropping image
        else if (requestCode == UCrop.REQUEST_CROP && resultCode == RESULT_OK) {
            if (data != null) {
                Uri uri = UCrop.getOutput(data);
                GalleryFragment galleryFragment = (GalleryFragment) getSupportFragmentManager().findFragmentByTag("galleryFragment");
                galleryFragment.addedImage(uri.getPath());
                // showImage(uri);
            }
        }
        //Succed choosing image from gallery
        else if (requestCode == PICK_IMAGE_GALLERY_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            Log.d("crush_test2","succeed choosing image");
            try {
                Uri sourceUri = data.getData();
                Log.d("crush_test2","sourceUri"+sourceUri);
                Uri destinationUri = Uri.fromFile(new File(problemNote, System.currentTimeMillis() + ""));
                if (aiFlag == true) {
                    Log.d("crush_test2","open AI crop activity");
                    openAiGalleryCropActivity(sourceUri, destinationUri);
                } else {
                    openCropActivity(sourceUri, destinationUri);
                }

            } catch (Exception e) {
                uiHelper.toast(this, "개발자에게 오류를 알려주세요");
            }
        }
    }

    private void openImagesDocument() {
        Log.d("crush_test2","openImagesDocument");
        Intent pictureIntent = new Intent(Intent.ACTION_GET_CONTENT);
        pictureIntent.setType("image/*");
        pictureIntent.addCategory(Intent.CATEGORY_OPENABLE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            String[] mimeTypes = new String[]{"image/jpeg", "image/png"};
            pictureIntent.putExtra(Intent.EXTRA_MIME_TYPES, mimeTypes);
        }
        startActivityForResult(Intent.createChooser(pictureIntent, "Select Picture"), PICK_IMAGE_GALLERY_REQUEST_CODE);
    }

    private void openCamera() {
        Intent pictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        File file;
        try {
            file = getImageFile(); // 1
        } catch (Exception e) {
            e.printStackTrace();
            uiHelper.toast(this, "Please take another image");
            return;
        }
        Uri uri;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) // 2
            uri = FileProvider.getUriForFile(this, BuildConfig.APPLICATION_ID.concat(".provider"), file);
        else
            uri = Uri.fromFile(file); // 3

        pictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, uri); // 4
        startActivityForResult(pictureIntent, CAMERA_ACTION_PICK_REQUEST_CODE);
    }

    private File getImageFile() throws IOException {
        String imageFileName = "JPEG_" + System.currentTimeMillis() + "_";
//        File storageDir = new File(
//                Environment.getExternalStoragePublicDirectory(
//                        Environment.DIRECTORY_DCIM
//                ), "Camera"
//        );
        File storageDir = this.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        Log.d("crush_test2",storageDir.getAbsolutePath());
        if (storageDir.exists())
            Log.d("crush_test2","File exists");
        else
            Log.d("crush_test2","File not exists");
        File file = File.createTempFile(
                imageFileName, ".jpg", storageDir
        );
//        currentPhotoPath = "file:" + file.getAbsolutePath();
        currentPhotoPath = "file:" + file.getAbsolutePath();
        Log.d("crush_test2",currentPhotoPath);
        return file;
    }

    private void openCropActivity(Uri sourceUri, Uri destinationUri) {
        UCrop.Options options = new UCrop.Options();
        options.setCircleDimmedLayer(true);
        options.setCropFrameColor(ContextCompat.getColor(this, R.color.colorAccent));
        UCrop.of(sourceUri, destinationUri)
                .start(this);

    }


    @Override
    public void folderClicked(String folderName) {
        problemNote = new File(dir.getAbsolutePath() + File.separator + folderName);

        ArrayList<String> imagePathList = new ArrayList<>();
//        ArrayList<String> solutionImageList = new ArrayList<>();
//        ArrayList<String> solutionTextList = new ArrayList<>();

        if (problemNote.exists()) {
            File[] allFiles = problemNote.listFiles(new FilenameFilter() {
                public boolean accept(File dir, String name) {

                    //return (name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png") || name.endsWith(".txt"));
                    return (name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png"));
                }
            });
            for (File f : allFiles) {

                //check whether file is solution
                String fileName = f.getName();
                if (fileName.contains("_solution")) {
                    continue;

                } else
                    imagePathList.add(f.getAbsolutePath());
            }

        }

        FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
        Log.d("check", "transcation start");
        Fragment galleryFragment = new GalleryFragment(imagePathList);
        Log.d("check", "galleryFragmentMade");
        transaction.replace(R.id.frame, galleryFragment, "galleryFragment");
        transaction.addToBackStack(null);
        transaction.commit();


    }

    private SharedPreferences sharedPreferences;

    @Override
    public void makeFolder(String folderName, int subjectId) {
        File subjectFolder = new File(dir.getPath(), folderName);
        subjectFolder.mkdirs();
        sharedPreferences = getSharedPreferences("folderIcons", MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        switch (subjectId) {
            case R.id.ic_folder:
                editor.putInt(folderName, R.drawable.ic_folder_48dp);
                break;
            case R.id.ic_history:
                editor.putInt(folderName, R.drawable.ic_history_48dp);
                break;
            case R.id.ic_cal:
                editor.putInt(folderName, R.drawable.ic_cal_48dp);
                break;
            case R.id.ic_law:
                editor.putInt(folderName, R.drawable.ic_law_48dp);
                break;
            case R.id.ic_science:
                editor.putInt(folderName, R.drawable.ic_science_48dp);
                break;
            case R.id.ic_writing:
                editor.putInt(folderName, R.drawable.ic_writing_48dp);
                break;
            case R.id.ic_symbols:
                editor.putInt(folderName, R.drawable.ic_symbols_48dp);
                break;
            case R.id.ic_pet:
                editor.putInt(folderName, R.drawable.ic_pet_48dp);
                break;
            case R.id.ic_school:
                editor.putInt(folderName, R.drawable.ic_school_48dp);
                break;
            case R.id.ic_idea:
                editor.putInt(folderName, R.drawable.ic_idea_48dp);
                break;
            case R.id.ic_art:
                editor.putInt(folderName, R.drawable.ic_art_48dp);
                break;
            case R.id.ic_heart:
                editor.putInt(folderName, R.drawable.ic_heart_48dp);
                break;
            default:
                editor.putInt(folderName, R.drawable.ic_folder_48dp);
                break;

        }
        editor.commit();

        FolderFragment folderFragment = (FolderFragment) getSupportFragmentManager().findFragmentByTag("folderFragment");
        folderFragment.makeFolder(folderName, subjectId);
    }

    @Override
    public void sendFolderData(String folderName, int subjectId) {
        this.folderName = folderName;
        this.subjectId = subjectId;
        makeFolder(folderName, subjectId);

    }

    //Auto Problem recognize section


    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = false;
    private Integer sensorOrientation = 90;


//    private Matrix frameToCropTransform;
//    private Matrix cropToFrameTransform;
//    //private MultiBoxTracker tracker;
//   // private OverlayView trackingOverlay;
//
//    protected int previewWidth = 0;
//    protected int previewHeight = 0;
//
//    private void initBox() {
//        previewHeight = TF_OD_API_INPUT_SIZE;
//        previewWidth = TF_OD_API_INPUT_SIZE;
//        frameToCropTransform =
//                ImageUtils.getTransformationMatrix(
//                        previewWidth, previewHeight,
//                        TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
//                        sensorOrientation, MAINTAIN_ASPECT);
//
//        cropToFrameTransform = new Matrix();
//        frameToCropTransform.invert(cropToFrameTransform);
//
////        tracker = new MultiBoxTracker(this);
////        trackingOverlay = findViewById(R.id.tracking_overlay);
////        trackingOverlay.addCallback(
////                canvas -> tracker.draw(canvas));
////
////        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);
//
//        try {
//            detector =
//                    YoloV4Classifier.create(
//                            getAssets(),
//                            TF_OD_API_MODEL_FILE,
//                            TF_OD_API_LABELS_FILE,
//                            TF_OD_API_IS_QUANTIZED);
//        } catch (final IOException e) {
//            e.printStackTrace();
//            LOGGER.e(e, "Exception initializing classifier!");
//            Toast toast =
//                    Toast.makeText(
//                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
//            toast.show();
//            finish();
//        }
//    }

    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results, Bitmap originalBitmap) {
        Log.d("test2", "handle Result Starts");
//        final Canvas canvas = new Canvas(bitmap);
//        final Paint paint = new Paint();
//        paint.setColor(Color.RED);
//        paint.setStyle(Paint.Style.STROKE);
//        paint.setStrokeWidth(2.0f);

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

        String name = String.valueOf(System.currentTimeMillis() / 1000);
        int crop_no = 0;
        GalleryFragment galleryFragment = (GalleryFragment) getSupportFragmentManager().findFragmentByTag("galleryFragment");


        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            Log.d("test12", "detection CLass" + result.getDetectedClass());
            Log.d("test12", "detection CLass" + result.getTitle());
            if (onlyWrong == true) {
                Log.d("test12", "OnlyWrongMode");
                if (result.getDetectedClass() == 0) {
                    Log.d("test12", "Skipped + " + result.getDetectedClass());
                    continue;
                }

            }
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
//                canvas.drawRect(location, paint);
//                Log.d("test2","left: "+location.left+"right: "+location.right+"bottom: "+location.bottom+"top: "+location.top);
//                int originalWidth= originalBitmap.getWidth(); int originalHeight=originalBitmap.getHeight();
//                int croppedWidth = bitmap.getWidth(); int croppedHeight = bitmap.getHeight();
//                float xRatio = originalWidth/croppedWidth; float yRatio=originalHeight/croppedHeight;
//                int left = (int)(location.left*xRatio); int right = (int)(location.right*xRatio);
//                int top = (int) (location.top*yRatio); int bottom = (int) (location.bottom*yRatio);

//                Bitmap problem = Bitmap.createBitmap(originalBitmap,
//                        left,
//                        top,
//                        right-left,
//                        bottom-top
//                );

//////TEST 10

                cropToFrameTransformations.mapRect(location);
                Log.d("test10", "frameToCrop - " + cropToFrameTransformations.toString());

                /////
                Bitmap problem = Bitmap.createBitmap(originalBitmap,
                        (int) location.left,
                        (int) location.top,
                        (int) (location.right) - (int) (location.left),
                        (int) (location.bottom) - (int) (location.top)
                );

                //make jpg file
                File savedImage = new File(problemNote, name + crop_no + ".jpg");
                Log.d("test2", savedImage.getPath());
                try {
                    FileOutputStream out = new FileOutputStream(savedImage);
                    problem.compress(Bitmap.CompressFormat.JPEG, 90, out);
                    out.flush();
                    out.close();
                    galleryFragment.addedImage(savedImage.getAbsolutePath());
                    Log.d("test2", "save Succeed!!");
                    crop_no++;
                } catch (Exception e) {
                    Log.d("test2", "fail to make file" + e.getMessage());
                }


//                cropToFrameTransform.mapRect(location);
//
//                result.setLocation(location);
//                mappedRecognitions.add(result);
            }
        }
        if(crop_no==0){
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "문제를 찾지 못했습니다. AI 민감도를 낮추거나 AI모드를 꺼서 직접 잘라주세요", Toast.LENGTH_SHORT);
            toast.show();
        }
//        tracker.trackResults(mappedRecognitions, new Random().nextInt());
//        trackingOverlay.postInvalidate();
        // imageView.setImageBitmap(bitmap);


    }

    public void changeFlag() {

        if (aiFlag == false)
            aiFlag = true;
        else
            aiFlag = false;
    }

    Matrix frameToCropTransformations;
    Matrix cropToFrameTransformations;

    private Bitmap rotateBitmap(Bitmap bitmap, int orientation) {

        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_NORMAL:
                return bitmap;
//            case 0:
//                return bitmap;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break;
//            case 180:
//                matrix.setRotate(180);
//                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setRotate(180);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break;
//            case 90:
//                matrix.setRotate(90);
//                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90);
                break;
//            case 270:
//                matrix.setRotate(-90);
//                break;
            default:
                return bitmap;
        }
        try {
            Bitmap bmRotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            bitmap.recycle();
            return bmRotated;
        } catch (OutOfMemoryError e) {
            e.printStackTrace();
            return null;
        }
    }

    public void openAiGalleryCropActivity(Uri resource, Uri destination) {
        Log.d("crush_test2","aiCropActivity Start");
        Log.d("crush_test2","exif Succeed");
        Cursor cursor = this.getContentResolver().query(resource,
                new String[] { MediaStore.Images.ImageColumns.ORIENTATION }, null, null, null);

        if (cursor.getCount() != 1) {
            Log.d("crush_test2","failed cursor");
        }

        cursor.moveToFirst();
        int orientation= cursor.getInt(0);
//        String picturePath = FileUtils.getPath(this,resource);
//        Log.d("crush_test2",picturePath);
//        Log.d("crush_test2","galleryStart");
//        try {
//            exif2 = new ExifInterface(picturePath);
//
//        } catch (Exception e) {
//            Log.d("crush_test2", "error"+ e.getMessage());
//        }
//        int orientation = exif2.getAttributeInt(ExifInterface.TAG_ORIENTATION,
//                ExifInterface.ORIENTATION_UNDEFINED);

        Log.d("crush_test2","orientation : "+orientation);

        Log.d("test2", "Ai Crop Activity starts");
        try {
            Log.d("gallery","start Image");
            Bitmap tempBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), resource);
            Bitmap sourceBitmap = rotateBitmap(tempBitmap, orientation);
//            Bitmap sourceBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), resource);
            Log.d("test2", "bitmap succeed :" + sourceBitmap.getHeight());

//            Bitmap cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);


            int image_height = sourceBitmap.getHeight();
            int image_width = sourceBitmap.getWidth();

            Bitmap cropBitmap = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);

            frameToCropTransformations = Utils.getTransformationMatrix(image_width, image_height, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, 0, false);
            Log.d("test10", "frameToCrop - " + frameToCropTransformations.toString());
            cropToFrameTransformations = new Matrix();
            Log.d("test10", "frameToCrop - " + cropToFrameTransformations.toString());
            frameToCropTransformations.invert(cropToFrameTransformations);
            Log.d("test10", "frameToCrop - " + frameToCropTransformations.toString());

            final Canvas canvas = new Canvas(cropBitmap);
            canvas.drawBitmap(sourceBitmap, frameToCropTransformations, null);

            Handler handler = new Handler();
            Log.d("test2", "made handler" + handler.toString());


            new Thread(() -> {
                final List<Classifier.Recognition> results = detector.recognizeImage(cropBitmap);
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        handleResult(cropBitmap, results, sourceBitmap);
                    }
                });
            }).start();

        } catch (Exception e) {
            Log.d("test2", e.getMessage());
        }


    }

    public void openAiCameraCropActivity(Uri resource, Uri destination) {
        ExifInterface exif = null;
        try {
            try {
                exif = new ExifInterface(resource.getPath());

            } catch (Exception e) {
                Log.d("camera_test", e.getMessage());
            }
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_UNDEFINED);

            Log.d("test2", "Ai Crop Activity starts");
            Bitmap tempBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), resource);
            Bitmap sourceBitmap = rotateBitmap(tempBitmap, orientation);
            Log.d("test2", "bitmap succeed :" + sourceBitmap.getHeight());

//            Bitmap cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);


            int image_height = sourceBitmap.getHeight();
            int image_width = sourceBitmap.getWidth();

            Bitmap cropBitmap = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);

            frameToCropTransformations = Utils.getTransformationMatrix(image_width, image_height, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, 0, false);
            Log.d("test10", "frameToCrop - " + frameToCropTransformations.toString());
            cropToFrameTransformations = new Matrix();
            Log.d("test10", "frameToCrop - " + cropToFrameTransformations.toString());
            frameToCropTransformations.invert(cropToFrameTransformations);
            Log.d("test10", "frameToCrop - " + frameToCropTransformations.toString());

            final Canvas canvas = new Canvas(cropBitmap);
            canvas.drawBitmap(sourceBitmap, frameToCropTransformations, null);

            Handler handler = new Handler();
            Log.d("test2", "made handler" + handler.toString());


            new Thread(() -> {
                final List<Classifier.Recognition> results = detector.recognizeImage(cropBitmap);
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        handleResult(cropBitmap, results, sourceBitmap);
                    }
                });
            }).start();

        } catch (Exception e) {
            Log.d("test2", e.getMessage());
        }


    }

    @Override
    public void onBackPressed() {
        if(fragmentPage==1){
            super.onBackPressed();
        }else{
            finish();
        }

    }

}