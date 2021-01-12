package com.yooong.studynotefragment;

import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textview.MaterialTextView;
import com.yalantis.ucrop.UCrop;
import com.yooong.studynotefragment.utils.OnSwipeTouchListener;
import com.yooong.studynotefragment.utils.UiHelper;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;

import static com.yooong.studynotefragment.MainActivity.CAMERA_ACTION_PICK_REQUEST_CODE;
import static com.yooong.studynotefragment.MainActivity.PICK_IMAGE_GALLERY_REQUEST_CODE;
import static com.yooong.studynotefragment.MainActivity.problemNote;

public class ProblemOpened extends AppCompatActivity {
    private LinearLayout layout;
    private Intent intent;
    private ArrayList<String> imageUriList;
    private ArrayList<String> solutionImagePathList;

    private SharedPreferences sharedPreferences;
    private int index;
    private int lastIndex;
    private ImageView problemOpened, solutionImage;
    private MaterialTextView solutionTextView,problemNumberView;
    private TextInputEditText editText;
    private MaterialButton saveTextSolutionButton, deleteTextSolutionButton, deleteImageSolutionButton;

    private File destinationFile;
    private Button gallery, camera, addTextButton;


    private String currentPhotoPath = "";

    private UiHelper uiHelper = new UiHelper();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_problem_opened);
        intent = getIntent();
        imageUriList = intent.getStringArrayListExtra("imageUriList");
        //solutionImagePathList = intent.getStringArrayListExtra("solutionImagePathList");
        solutionImagePathList = new ArrayList<>();


        //@TODO change solution list read from here


        File[] allFiles = problemNote.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {

                //return (name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png") || name.endsWith(".txt"));
                return (name.endsWith("solution.jpg"));
            }
        });
        for (File f : allFiles) {

            solutionImagePathList.add(f.getAbsolutePath());
        }


        //
        lastIndex = imageUriList.size() - 1;
        index = intent.getIntExtra("index", 0);
        layout = findViewById(R.id.linearLayout);

        solutionTextView = findViewById(R.id.viewTextSolution);
        editText = findViewById(R.id.writeTextSolution);
        saveTextSolutionButton = findViewById(R.id.saveTextSolution);
        deleteTextSolutionButton = findViewById(R.id.deleteTextSolution);
        deleteImageSolutionButton = findViewById(R.id.deleteImageSolution);
        solutionImage = findViewById(R.id.imageSolution);
        problemNumberView = findViewById(R.id.problemNoTextView);

        problemNumberView.setText("문제번호"+(index+1));
        Log.d("test11", "OnCreate");

        sharedPreferences = getSharedPreferences(problemNote.getName(), MODE_PRIVATE);

        //text initialize
        Log.d("test9", "textInitialLized");
        showTextSolution(imageUriList.get(index));

        //show current Image
        String currentImage = imageUriList.get(index);
        problemOpened = findViewById(R.id.problemOpened);
        Glide
                .with(this)
                .load(currentImage)
                .into(problemOpened);

        solutionExists(currentImage);

        //Button Actions
        gallery = findViewById(R.id.addFromGalleryS);
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //openGallery();
                openImagesDocument();
            }
        });

        camera = findViewById(R.id.addFromCameraS);
        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openCamera();
            }
        });

        addTextButton = findViewById(R.id.addText);
        addTextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                editSolutionText(imageUriList.get(index));
            }
        });

        saveTextSolutionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveSolutionText(imageUriList.get(index));
            }
        });

        deleteTextSolutionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                deleteSolutionText(imageUriList.get(index));
            }
        });

        deleteImageSolutionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                deleteSolutionExists(imageUriList.get(index));
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "삭제되었습니다", Toast.LENGTH_SHORT);
                toast.show();
                deleteImageSolutionButton.setVisibility(View.INVISIBLE);
                solutionImage.setVisibility(View.GONE);
            }
        });

        //Swipe actions
        layout.setOnTouchListener(new OnSwipeTouchListener(ProblemOpened.this) {
            @Override
            public void onSwipeRight() {
                String nextImagePath;
                super.onSwipeRight();
                if (index == 0) {
                    index = lastIndex;
                    nextImagePath = imageUriList.get(index);
                    Glide.with(getApplicationContext()).load(nextImagePath).into(problemOpened);
                    showTextSolution(nextImagePath);
                } else {
                    index--;
                    nextImagePath = imageUriList.get(index);
                    Glide.with(getApplicationContext()).load(nextImagePath).into(problemOpened);
                    showTextSolution(nextImagePath);
                }
                problemNumberView.setText("문제번호"+(index+1));
                solutionExists(nextImagePath);

            }

            @Override
            public void onSwipeLeft() {
                super.onSwipeLeft();
                String nextImagePath;
                Log.d("test4", "swipeLeft");
                if (index == lastIndex) {
                    index = 0;
                    nextImagePath = imageUriList.get(index);
                    Glide.with(getApplicationContext()).load(nextImagePath).into(problemOpened);
                    showTextSolution(nextImagePath);
                } else {
                    index++;
                    nextImagePath = imageUriList.get(index);
                    Glide.with(getApplicationContext()).load(nextImagePath).into(problemOpened);
                    showTextSolution(nextImagePath);
                }
                problemNumberView.setText("문제번호"+(index+1));
                solutionExists(nextImagePath);

            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d("test11", "OnResume");
    }

    //SaveTextLogics

    private void showTextSolution(String imagePath) {
        String text = sharedPreferences.getString(imagePath, null);
        Log.d("test9", "show text  :  " + text);
        if (text != null && !text.equals("")) {
            Log.d("test9", "text not null, show solution TextView");
            solutionTextView.setVisibility(View.VISIBLE);
            solutionTextView.setText(text);
        } else {
            Log.d("test9", "No Text Exists, hide solutionTextview");
            solutionTextView.setText("");
            solutionTextView.setVisibility(View.GONE);
        }
        Log.d("test9", "hideEditTextAnyway");
        editText.setVisibility(View.GONE);
    }

    private void editSolutionText(String imagePath) {
        Log.d("test9", "editSolutionText Called");
        String text = sharedPreferences.getString(imagePath, null);
        Log.d("test9", "get text from shared Preference" + text);
        solutionTextView.setVisibility(View.GONE);
        Log.d("test9", "hideded solutionTextView and show editText");
        editText.setVisibility(View.VISIBLE);
        if (text != null && !text.equals("")) {
            Log.d("test9", "text exists");
            editText.setText(text);
        }
        Log.d("test9", "show editText");
        saveTextSolutionButton.setVisibility(View.VISIBLE);
        deleteTextSolutionButton.setVisibility(View.VISIBLE);
    }

    private void saveSolutionText(String imagePath) {
        Log.d("test9", "saveSolutionCalled");

        String text = editText.getText().toString();
        Log.d("test9", "get Texts in editText :" + text);
        if (!text.equals("") && text != null) {
            Log.d("test9", "something written");
            solutionTextView.setText(editText.getText());
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString(imagePath, editText.getText().toString());
            editor.commit();
            Log.d("test9", "saved data, show soltutiontextView");

        } else {
            Log.d("test9", "nothing written");
            Toast.makeText(this, "풀이를 입력하세요", Toast.LENGTH_SHORT).show();
        }
        Log.d("test9", "set Editor text to nothing and hide, show ");
        solutionTextView.setVisibility(View.VISIBLE);
        editText.setText("");
        editText.setVisibility(View.GONE);
        saveTextSolutionButton.setVisibility(View.GONE);
        deleteTextSolutionButton.setVisibility(View.GONE);


    }

    private void deleteSolutionText(String imagePath) {
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.remove(imagePath);
        editor.commit();
        editText.setText("");
        editText.setVisibility(View.GONE);
        saveTextSolutionButton.setVisibility(View.GONE);
        deleteTextSolutionButton.setVisibility(View.GONE);
    }
    //Camera & album logics

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // Succed taking picture from Camera
        if (requestCode == CAMERA_ACTION_PICK_REQUEST_CODE && resultCode == RESULT_OK) {
            Uri uri = Uri.parse(currentPhotoPath);
//            Uri test = Uri.fromFile(new File(problemNote,""+System.currentTimeMillis()));
//            Log.d("test",test.getPath());
            destinationFile = new File(imageUriList.get(index).replace(".jpg", "_solution.jpg"));
            Log.d("test3", destinationFile.getAbsolutePath());
            openCropActivity(uri, Uri.fromFile(destinationFile));

        }
        //Succeed cropping image
        else if (requestCode == UCrop.REQUEST_CROP && resultCode == RESULT_OK) {
            if (data != null) {
                Uri uri = UCrop.getOutput(data);
                Log.d("test4", uri.getPath());
                solutionImagePathList.add(uri.getPath());
                //solutionImagePathList.notify();
                Glide
                        .with(this)
                        .load(uri.getPath())
                        .into(solutionImage);
                solutionImage.setVisibility(View.VISIBLE);
                deleteImageSolutionButton.setVisibility(View.VISIBLE);
            }
        }
        //Succed choosing image from gallery
        else if (requestCode == PICK_IMAGE_GALLERY_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            try {
                Log.d("test3", "시작");
                Uri sourceUri = data.getData();
                Log.d("test3", "Uri : " + sourceUri.toString());
                String matchImagePath = imageUriList.get(index);
                Log.d("test3", "imageUri : " + matchImagePath);
                destinationFile = new File(matchImagePath.substring(0, matchImagePath.length() - 4) + "_solution" + ".jpg");
                Log.d("test3", "파일 생성 완료 : " + destinationFile.getAbsolutePath());
                Log.d("test3", destinationFile.getAbsolutePath());
                openCropActivity(sourceUri, Uri.fromFile(destinationFile));

            } catch (Exception e) {
                uiHelper.toast(this, "오류는 갤러리에서 발생");
            }
        }
    }

    private File getImageFile() throws IOException {
        String imageFileName = "JPEG_" + System.currentTimeMillis() + "_";
        File storageDir = new File(
                Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_DCIM
                ), "Camera"
        );
        System.out.println(storageDir.getAbsolutePath());
        if (storageDir.exists())
            System.out.println("File exists");
        else
            System.out.println("File not exists");
        File file = File.createTempFile(
                imageFileName, ".jpg", storageDir
        );
        currentPhotoPath = "file:" + file.getAbsolutePath();
        return file;
    }

    private void openImagesDocument() {
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

    private void openCropActivity(Uri sourceUri, Uri destinationUri) {

        UCrop.Options options = new UCrop.Options();
        options.setCircleDimmedLayer(true);
        options.setCropFrameColor(ContextCompat.getColor(this, R.color.colorAccent));
        UCrop.of(sourceUri, destinationUri)
                .start(this);

    }

    private final int textSolutionExists = 1001;
    private final int imageSolutionExists = 1002;
    private final int noSolutionExists = 1004;

    // Solution Check and add logics
    private void solutionExists(String filePath) {
        String[] filePaths = filePath.split("/");
        String fileName = filePaths[filePaths.length - 1];
        Log.d("test4", fileName);

        fileName = fileName.substring(0, fileName.length() - 4);
        Log.d("test4", fileName);


        //do they notice new data added?
        Log.d("test4", "solutionImagePathList length" + solutionImagePathList.size());
        for (String solutionImagePath : solutionImagePathList) {
            if (solutionImagePath.contains(fileName)) {
                Log.d("test4", "solution Image Found : " + solutionImagePath);
                Glide
                        .with(this)
                        .load(solutionImagePath)
                        .into(solutionImage);
                Log.d("test4", "glide Succeed, button visible");
                solutionImage.setVisibility(View.VISIBLE);
                deleteImageSolutionButton.setVisibility(View.VISIBLE);
                break;
            } else {
                Log.d("test4", "not match : " + solutionImagePath);
            }
            solutionImage.setVisibility(View.GONE);
            deleteImageSolutionButton.setVisibility(View.GONE);
        }


    }

    private void deleteSolutionExists(String filePath) {
        String[] filePaths = filePath.split("/");
        String fileName = filePaths[filePaths.length - 1];
        Log.d("test4", fileName);

        fileName = fileName.substring(0, fileName.length() - 4);
        Log.d("test4", fileName);

        for (String solutionImagePath : solutionImagePathList) {
            if (solutionImagePath.contains(fileName)) {
                Log.d("test11", "solution Image Found : delete : " + solutionImagePath);
                File deleteFile = new File(solutionImagePath);
                boolean succeed = deleteFile.delete();
                Log.d("test11", "status : " + succeed);
                boolean remove = solutionImagePathList.remove(solutionImagePath);
                Log.d("test11", "deleted from List?  " + remove);
            } else {
                Log.d("test11", "not match : " + solutionImagePath);
            }
        }


    }
}