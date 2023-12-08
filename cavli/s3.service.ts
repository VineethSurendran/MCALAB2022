import { Injectable } from '@angular/core';
import { S3 } from 'aws-sdk';

@Injectable({
 providedIn: 'root'
})
export class S3Service {
 private s3 = new S3();

 async uploadFile(file: File): Promise<any> {
    const params = {
      Bucket: 'your-bucket-name',
      Key: file.name,
      Body: file,
      ACL: 'public-read'
    };

    try {
      const result = await this.s3.upload(params).promise();
      return { status: 200, data: result };
    } catch (error) {
      return { status: 500, data: error };
    }
 }


}
