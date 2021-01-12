package com.yooong.studynotefragment.recyclerView;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.yooong.studynotefragment.R;

import java.util.ArrayList;

import static android.content.Context.MODE_PRIVATE;

public class folderAdapter extends RecyclerView.Adapter<folderAdapter.folderViewHolder> {

    private ArrayList<String> folderNames;
    private String dirPath;
    private int iconId;
    private OnFolderClickListener onFolderClickListener;
    private SharedPreferences sharedPreferences;



    //interface to handle data communication
    public interface OnFolderClickListener {
        void onFolderClicked(String folderName);
        void onDeleteButtonClicked(String folderName);
    }


    // Provide a suitable constructor (depends on the kind of dataset)
    public folderAdapter(Context context,ArrayList<String> folderNames,String dirPath,OnFolderClickListener onFolderClickListener) {
        this.sharedPreferences = context.getSharedPreferences("folderIcons",MODE_PRIVATE);
        this.folderNames = folderNames;
        this.dirPath = dirPath;
        this.onFolderClickListener=onFolderClickListener;
    }

    // Provide a reference to the views for each data item
    // Complex data items may need more than one view per item, and
    // you provide access to all the views for a data item in a view holder
    public static class folderViewHolder extends RecyclerView.ViewHolder {
        // each data item is just a string in this case
        public TextView textView;
        public ImageView imageView;
        public MaterialCardView materialCard;
        public MaterialButton deleteButton;

        public folderViewHolder(View itemView) {
            super(itemView);
            materialCard=itemView.findViewById(R.id.folderCard);
            textView = itemView.findViewById(R.id.folderName);
            imageView = itemView.findViewById(R.id.folderImage);
            deleteButton = itemView.findViewById(R.id.deleteFolderButton);

        }
    }

    // Create new views (invoked by the layout manager)
    @Override
    public folderAdapter.folderViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {

        Context context = parent.getContext();
        LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) ;

        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.folder_view, parent, false) ;
        folderAdapter.folderViewHolder vh = new folderAdapter.folderViewHolder(view) ;

        return vh ;
    }

    // onBindViewHolder() - position에 해당하는 데이터를 뷰홀더의 아이템뷰에 표시.
    @Override
    public void onBindViewHolder(folderAdapter.folderViewHolder holder, int position) {

        //shouldChangeto default  = folder
        iconId = sharedPreferences.getInt(folderNames.get(position),R.drawable.ic_folder_48dp);
        Glide.with(holder.itemView.getContext()).load(iconId).into(holder.imageView);

        String text = folderNames.get(position) ;
        holder.textView.setText(text) ;
        holder.materialCard.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                Log.d("test7","Long Clicked");
                holder.deleteButton.setVisibility(View.VISIBLE);
                return true;
            }
        });
        holder.materialCard.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                TextView tv = (TextView)v.findViewById(R.id.folderName);
                String folderName = tv.getText().toString();
                Log.d("test7","clicked");

                onFolderClickListener.onFolderClicked(folderName);
                //MainActivity.openFolder(folderName);
//                Intent intent = new Intent(mContext,ProblemNote.class);
//                intent.putExtra("notePath",dirPath+ File.separator+name);
//                mContext.startActivity(intent);

            }
        });

        holder.deleteButton.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
                Log.d("test6","delete button clicked, send data = "+folderNames.get(position));
                onFolderClickListener.onDeleteButtonClicked(folderNames.get(position));



            }
        });
    }

    // getItemCount() - 전체 데이터 갯수 리턴.
    @Override
    public int getItemCount() {
        return folderNames.size() ;
    }

}
